import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

from model import Unet
from noise_scheduler import LinearNoiseScheduler

NUM_TIMESTEPS = 1000
BETA_START = 0.0001
BETA_END = 0.02
IMAGE_SIZE = 28
IN_CHANNELS = 1


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_mnist_dataloaders(data_dir: str, batch_size: int):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader


def train_epoch(model, train_loader, scheduler, optimizer, device, epoch, total_epochs):
    model.train()
    total_loss = 0.0

    progress = tqdm(train_loader, desc=f"Train {epoch}/{total_epochs}", leave=False)
    for images, _ in progress:
        images = images.to(device)
        batch_size = images.shape[0]

        noise = torch.randn_like(images)
        timesteps = torch.randint(0, scheduler.num_timesteps, (batch_size,), device=device)
        noisy_images = scheduler.add_noise(images, noise, timesteps)

        noise_pred = model(noisy_images, timesteps)
        loss = F.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(train_loader)


@torch.no_grad()
def evaluate(model, test_loader, scheduler, device, epoch, total_epochs):
    model.eval()
    total_loss = 0.0

    progress = tqdm(test_loader, desc=f"Eval {epoch}/{total_epochs}", leave=False)
    for images, _ in progress:
        images = images.to(device)
        batch_size = images.shape[0]

        noise = torch.randn_like(images)
        timesteps = torch.randint(0, scheduler.num_timesteps, (batch_size,), device=device)
        noisy_images = scheduler.add_noise(images, noise, timesteps)

        noise_pred = model(noisy_images, timesteps)
        loss = F.mse_loss(noise_pred, noise)
        total_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(test_loader)


@torch.no_grad()
def sample_images(model, scheduler, device, num_images: int, output_path: Path):
    model.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    images = torch.randn(num_images, IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE, device=device)

    for t in tqdm(
        reversed(range(scheduler.num_timesteps)),
        desc="Sampling",
        leave=False,
        total=scheduler.num_timesteps,
    ):
        timesteps = torch.full((num_images,), t, device=device, dtype=torch.long)
        noise_pred = model(images, timesteps)
        images, _ = scheduler.sample_prev_timestep(images, noise_pred, t)

    images = (images.clamp(-1, 1) + 1) / 2
    save_image(images, output_path, nrow=int(num_images ** 0.5))


def save_checkpoint(model, optimizer, epoch, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        output_dir / "checkpoint.pt",
    )


def load_checkpoint(model, optimizer, checkpoint_path: Path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train a diffusion model on MNIST")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--sample-only", action="store_true", help="Only generate samples from a checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    output_dir = Path(args.output_dir)

    model = Unet(in_channels=IN_CHANNELS).to(device)
    scheduler = LinearNoiseScheduler(NUM_TIMESTEPS, BETA_START, BETA_END).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    if args.checkpoint:
        start_epoch = load_checkpoint(model, optimizer, Path(args.checkpoint), device)
        print(f"Loaded checkpoint from epoch {start_epoch}")

    if args.sample_only:
        sample_images(
            model,
            scheduler,
            device,
            args.num_samples,
            output_dir / "samples.png",
        )
        print(f"Saved samples to {output_dir / 'samples.png'}")
        return

    print(f"Using device: {device}")
    train_loader, test_loader = get_mnist_dataloaders(args.data_dir, args.batch_size)
    print(f"Loaded MNIST: {len(train_loader.dataset)} train, {len(test_loader.dataset)} test images")

    for epoch in range(start_epoch, args.epochs):
        epoch_num = epoch + 1
        train_loss = train_epoch(
            model, train_loader, scheduler, optimizer, device, epoch_num, args.epochs
        )
        test_loss = evaluate(
            model, test_loader, scheduler, device, epoch_num, args.epochs
        )

        tqdm.write(
            f"Epoch {epoch_num}/{args.epochs} | "
            f"train loss: {train_loss:.4f} | test loss: {test_loss:.4f}"
        )

        save_checkpoint(model, optimizer, epoch + 1, output_dir)
        sample_images(
            model,
            scheduler,
            device,
            args.num_samples,
            output_dir / f"samples_epoch_{epoch + 1}.png",
        )

    print(f"Training complete. Checkpoints and samples saved to {output_dir}")


if __name__ == "__main__":
    main()
