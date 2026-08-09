import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from model_large import UnetLarge
from noise_scheduler import LinearNoiseScheduler

NUM_TIMESTEPS = 1000
BETA_START = 0.0001
BETA_END = 0.02
IMAGE_SIZE = 128
IN_CHANNELS = 3


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def use_bf16(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.is_bf16_supported()


MIN_CELEBA_IMAGES = 200_000


def count_celeba_images(celeba_dir: Path) -> int:
    image_dir = celeba_dir / "img_align_celeba"
    if not image_dir.is_dir():
        return 0
    return sum(1 for _ in image_dir.glob("*.jpg"))


def normalize_filename(name: str) -> str:
    name = name.strip()
    if name.endswith(".jpg"):
        return name
    if name.isdigit():
        return f"{int(name):06d}.jpg"
    return name


def partition_file_exists(celeba_dir: Path) -> bool:
    return (celeba_dir / "list_eval_partition.txt").exists() or (
        celeba_dir / "list_eval_partition.csv"
    ).exists()


def celeba_is_ready(data_dir: str) -> bool:
    celeba_dir = Path(data_dir) / "celeba"
    if not (celeba_dir / "img_align_celeba").is_dir():
        return False
    if not partition_file_exists(celeba_dir):
        return False
    return count_celeba_images(celeba_dir) >= MIN_CELEBA_IMAGES


def load_partition(celeba_dir: Path) -> dict[str, int]:
    txt_path = celeba_dir / "list_eval_partition.txt"
    csv_path = celeba_dir / "list_eval_partition.csv"
    partition: dict[str, int] = {}

    if txt_path.exists():
        with txt_path.open(encoding="utf-8") as handle:
            handle.readline()
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                filename, split_value = line.rsplit(" ", 1)
                partition[normalize_filename(filename)] = int(split_value)
    elif csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            next(reader)
            for row in reader:
                if row:
                    partition[normalize_filename(row[0])] = int(row[1])
    else:
        raise FileNotFoundError(f"Missing partition file in {celeba_dir}")

    return partition


class CelebAImageDataset(Dataset):
    def __init__(self, data_dir: str, split: str, transform=None):
        celeba_dir = Path(data_dir) / "celeba"
        self.image_dir = celeba_dir / "img_align_celeba"
        self.transform = transform
        split_ids = {"train": 0, "valid": 1, "test": 2}
        if split not in split_ids:
            raise ValueError(f"Unknown split: {split}")

        partition = load_partition(celeba_dir)
        target_split = split_ids[split]
        self.filenames = sorted(
            path.name
            for path in self.image_dir.glob("*.jpg")
            if partition.get(path.name) == target_split
        )
        if not self.filenames:
            raise RuntimeError(f"No CelebA images found for split={split} under {self.image_dir}")

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int):
        image_path = self.image_dir / self.filenames[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, 0


def get_celeba_dataloaders(data_dir: str, batch_size: int, num_workers: int):
    if not celeba_is_ready(data_dir):
        raise RuntimeError(
            f"CelebA not found under {Path(data_dir) / 'celeba'}. "
            "Extract the dataset first (see train_celeb_colab.ipynb section 4)."
        )

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = CelebAImageDataset(data_dir, split="train", transform=transform)
    test_dataset = CelebAImageDataset(data_dir, split="valid", transform=eval_transform)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_loader, test_loader


def train_epoch(model, train_loader, scheduler, optimizer, device, epoch, total_epochs, amp_enabled):
    model.train()
    total_loss = 0.0
    progress = tqdm(train_loader, desc=f"Train {epoch}/{total_epochs}", leave=False)

    for images, _ in progress:
        images = images.to(device, non_blocking=True)
        batch_size = images.shape[0]

        noise = torch.randn_like(images)
        timesteps = torch.randint(0, scheduler.num_timesteps, (batch_size,), device=device)
        noisy_images = scheduler.add_noise(images, noise, timesteps)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
            noise_pred = model(noisy_images, timesteps)
            loss = F.mse_loss(noise_pred, noise)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(train_loader)


@torch.no_grad()
def evaluate(model, test_loader, scheduler, device, epoch, total_epochs, amp_enabled):
    model.eval()
    total_loss = 0.0
    progress = tqdm(test_loader, desc=f"Eval {epoch}/{total_epochs}", leave=False)

    for images, _ in progress:
        images = images.to(device, non_blocking=True)
        batch_size = images.shape[0]

        noise = torch.randn_like(images)
        timesteps = torch.randint(0, scheduler.num_timesteps, (batch_size,), device=device)
        noisy_images = scheduler.add_noise(images, noise, timesteps)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
            noise_pred = model(noisy_images, timesteps)
            loss = F.mse_loss(noise_pred, noise)

        total_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(test_loader)


@torch.no_grad()
def sample_images(
    model,
    scheduler,
    device,
    num_images: int,
    output_path: Path,
    sample_steps: int,
    amp_enabled: bool,
):
    model.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    images = torch.randn(num_images, IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE, device=device)
    ddim_timesteps = scheduler.get_ddim_timesteps(sample_steps, device=device)

    for i in tqdm(range(len(ddim_timesteps) - 1), desc="DDIM sampling", leave=False):
        t = ddim_timesteps[i].item()
        t_prev = ddim_timesteps[i + 1].item()
        t_batch = torch.full((num_images,), t, device=device, dtype=torch.long)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
            noise_pred = model(images, t_batch)
        images, _ = scheduler.ddim_step(images, noise_pred.float(), t, t_prev, eta=0.0)

    images = (images.clamp(-1, 1) + 1) / 2
    save_image(images, output_path, nrow=int(num_images ** 0.5))


def save_checkpoint(model, optimizer, scaler, epoch, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        },
        output_dir / "checkpoint.pt",
    )


def load_checkpoint(model, optimizer, scaler, checkpoint_path: Path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return checkpoint["epoch"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train a larger diffusion model on CelebA 128x128")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./outputs_celeb")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--sample-steps", type=int, default=50, help="DDIM denoising steps for preview images")
    parser.add_argument("--num-res-blocks", type=int, default=2, help="ResNet blocks per down/up stage")
    parser.add_argument("--num-mid-blocks", type=int, default=2, help="Bottleneck blocks at lowest resolution")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--sample-only", action="store_true")
    parser.add_argument("--no-amp", action="store_true", help="Disable bf16 autocast even on CUDA")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    output_dir = Path(args.output_dir)
    amp_enabled = use_bf16(device) and not args.no_amp

    model = UnetLarge(
        in_channels=IN_CHANNELS,
        num_res_blocks=args.num_res_blocks,
        num_mid_blocks=args.num_mid_blocks,
    ).to(device)
    scheduler = LinearNoiseScheduler(NUM_TIMESTEPS, BETA_START, BETA_END).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = None

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Using device: {device}")
    print(f"Model parameters: {param_count:,}")
    print(f"bf16 autocast: {amp_enabled}")

    start_epoch = 0
    if args.checkpoint:
        start_epoch = load_checkpoint(model, optimizer, scaler, Path(args.checkpoint), device)
        print(f"Loaded checkpoint from epoch {start_epoch}")

    if args.sample_only:
        sample_images(
            model,
            scheduler,
            device,
            args.num_samples,
            output_dir / "samples.png",
            args.sample_steps,
            amp_enabled,
        )
        print(f"Saved samples to {output_dir / 'samples.png'}")
        return

    train_loader, test_loader = get_celeba_dataloaders(
        args.data_dir,
        args.batch_size,
        args.num_workers,
    )
    print(
        f"Loaded CelebA: {len(train_loader.dataset)} train, "
        f"{len(test_loader.dataset)} valid images at {IMAGE_SIZE}x{IMAGE_SIZE}"
    )

    for epoch in range(start_epoch, args.epochs):
        epoch_num = epoch + 1
        train_loss = train_epoch(
            model, train_loader, scheduler, optimizer, device, epoch_num, args.epochs, amp_enabled
        )
        test_loss = evaluate(
            model, test_loader, scheduler, device, epoch_num, args.epochs, amp_enabled
        )

        tqdm.write(
            f"Epoch {epoch_num}/{args.epochs} | "
            f"train loss: {train_loss:.4f} | valid loss: {test_loss:.4f}"
        )

        save_checkpoint(model, optimizer, scaler, epoch_num, output_dir)
        sample_images(
            model,
            scheduler,
            device,
            args.num_samples,
            output_dir / f"samples_epoch_{epoch_num}.png",
            args.sample_steps,
            amp_enabled,
        )

    print(f"Training complete. Checkpoints and samples saved to {output_dir}")


if __name__ == "__main__":
    main()
