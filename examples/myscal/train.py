from torch import optim
from torch.nn.functional import binary_cross_entropy
from utils import *
from dataset import *
import torch
import numpy as np
from model import *
import scallopy
import matplotlib.pyplot as plt
from datetime import datetime
import math

print(torch.cuda.is_available())
print(torch.version.cuda)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = HWF(3).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.BCELoss()

np.random.seed(1234)
torch.manual_seed(1234)
train_set = MathExprDataset('train', numSamples=int(10000 * 1.00),
                            randomSeed=777)
test_set = MathExprDataset('test')
print('train:', len(train_set), '  test:', len(test_set))

batch_size = 16
num_workers = 0

train_dataloader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=num_workers,
                                               collate_fn=MathExpr_collate)
eval_dataloader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=num_workers,
                                              collate_fn=MathExpr_collate)

count = 0

epoch_losses = []
epoch_accuracies = []

num_epochs = 100

log_file = "training_log.txt"
with open(log_file, "w") as f:
    f.write("Epoch, Average Loss, Accuracy (%)\n")

for epoch in range(1, num_epochs + 1):
    print("epoch: ", epoch)
    model.train()

    train_loss = 0.0
    total_correct = 0
    total_samples = 0
    batch_count = 0
    count = 0
    for sample in train_dataloader:
        print("count: ", count + 1)
        count += 1
        img_seq = sample['img_seq']
        res = sample['res']
        seq_len = sample['len']
#
        optimizer.zero_grad()
        (results, probs) = model(img_seq.to(device), seq_len.to(device))
        threshold = 0.01

        batch_size, num_candidates = probs.shape

        # Construct target tensor:
        # For each sample, traverse the results list,
        # If the absolute difference between the true value and
        # the candidate value is less than the threshold,
        # it is considered correct (1.0), otherwise it is incorrect (0.0)
        target = torch.tensor(
            [
                [1.0 if abs(res[i].item() - candidate) < threshold else 0.0 for
                 candidate in results]
                for i in range(batch_size)
            ],
            dtype=torch.float32
        ).to(device)

        # Calculate losses
        loss = loss_fn(probs.to(device), target.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if not math.isnan(loss.item()):
            train_loss += loss.item()

        # Compute the accuracy
        if num_candidates > 0:
            pred_indices = torch.argmax(probs, dim=1).to(device)
            target_indices = torch.argmax(target, dim=1).to(device)
            batch_correct = torch.sum(
                torch.where(torch.sum(target.to(device), dim=1) > 0,
                            pred_indices.to(device) == target_indices.to(device),
                            torch.zeros(batch_size, dtype=torch.bool).to(device))
            ).item()
        else:
            batch_correct = 0

        total_correct += batch_correct
        total_samples += batch_size

    # Compute the average loss in an epoch
    avg_loss = train_loss / batch_count if batch_count > 0 else 0.0
    accuracy = 100.0 * total_correct / total_samples if total_samples > 0 else 0.0
    epoch_losses.append(avg_loss)
    epoch_accuracies.append(accuracy)

    epoch_info = f"Epoch {epoch}: Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%"
    print(epoch_info)

    # Write information to log file
    with open(log_file, "a") as f:
        f.write(epoch_info + "\n")


# Draw a training curve chart (Loss and Accuracy)
epochs = range(1, num_epochs + 1)
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs, epoch_losses, marker='o', label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, epoch_accuracies, marker='o', color='orange', label='Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training Accuracy')
plt.legend()

plt.tight_layout()

curve_path = "training_curves.png"
plt.savefig(curve_path)
plt.close()

print(f"Training curves saved to {curve_path}")
