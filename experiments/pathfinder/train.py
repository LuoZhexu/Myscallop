import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image
import os

from pathfinderdataset import PathfinderDataset
from model import NeuroSymbolicPathfinderModel

cuda_num = 4
top_k = 7
def train_one_epoch(model, device, train_loader, optimizer, loss_fn, epoch_idx):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        valid_mask = (labels != -1)
        if valid_mask.sum() == 0:
            continue
        images = images[valid_mask]
        labels = labels[valid_mask]
        gt = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()

        mapping, probs = model(images)

        loss = loss_fn(probs, gt)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = (probs > 0.5).float()
        correct += (preds == gt).sum().item()
        total += images.size(0)

        print(
            f"[Epoch {epoch_idx + 1}] Batch {batch_idx + 1}/{len(train_loader)} | Loss: {loss.item():.4f}")

    epoch_loss = running_loss / total
    accuracy = correct / total
    return epoch_loss, accuracy


def evaluate(model, device, val_loader, loss_fn):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            gt = labels.float().unsqueeze(1).to(device)
            mapping, probs = model(images)
            loss = loss_fn(probs, gt)
            running_loss += loss.item() * images.size(0)
            preds = (probs > 0.5).float()
            correct += (preds == gt).sum().item()
            total += images.size(0)
    epoch_loss = running_loss / total
    accuracy = correct / total
    return epoch_loss, accuracy

def main():
    device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    log_path = os.path.join(os.path.dirname(__file__), "train_log.txt")

    # preprocess the data
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # load the dataset
    dataset = PathfinderDataset(
        root_dir='/home/luozx/scallop/experiments/pathfinder/dataset_generation/pathfinder32_data/pathfinder32',
        subdir='curv_baseline',
        transform=transform
    )

    # 90% train，10% eval
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset,
                                                       [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True,
                              num_workers=0)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False,
                            num_workers=0)

    # e.g. img_size=32, output_dim=256, provenance="difftopbottomkclauses", k=7
    model = NeuroSymbolicPathfinderModel(img_size=32, output_dim=256,
                                         provenance="difftopbottomkclauses",
                                         k=top_k)
    model = model.to(device)
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    num_epochs = 100
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, device, train_loader,
                                                optimizer, loss_fn, epoch)
        val_loss, val_acc = evaluate(model, device, val_loader, loss_fn)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        log_str = (f"Epoch {epoch + 1}/{num_epochs}: "
                   f"Train loss={train_loss:.4f}, acc={train_acc:.4f}; "
                   f"Val loss={val_loss:.4f}, acc={val_acc:.4f}")
        print(log_str)

        with open(os.path.join(os.path.dirname(__file__), f"train_log{cuda_num}_{top_k}.txt"),
                  "a") as f:
            f.write(log_str + "\n")
    # Plot the curves
    epochs = np.arange(1, num_epochs + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label='Train Accuracy')
    plt.plot(epochs, val_accs, label='Val Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epoch")
    plt.legend()

    plot_path = "/home/luozx/scallop/experiments/pathfinder/train_plot.png"
    plt.savefig(plot_path)
    print(f"[INFO] Curves saved to {plot_path}")


if __name__ == "__main__":
    main()
