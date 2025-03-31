import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, optim
import torchvision
from PIL import Image
import random
import scallopy


class NeuralNet(nn.Module):
    def __init__(self, img_size=32, output_dim=16 + 15 * 16):
        """
        Parameters:
        img_size:  Enter image size (32 or 128)
        output_dim:  The dimension of the output feature vector represents the sum of the dot and dash numbers in the grid
        (default set to 100, can be set according to the task)
        Architecture Description:
        -4-layer CNN is used to extract image features
        -After flattening, use 2 layers of MLP to map 4096 dimensional features to the output dimension
        -Each output undergoes Sigmoid transformation to obtain the probability of [0,1],
        representing the possibility of the corresponding "dot" or "dash" existing
        """
        super(NeuralNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3,
                               padding=1)  # [32, img_size, img_size]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3,
                               padding=1)  # [64, img_size, img_size]
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3,
                               padding=1)  # [128, img_size, img_size]
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3,
                               padding=1)  # [256, img_size, img_size]
        self.pool = nn.MaxPool2d(2, 2)

        flattened_dim = 256 * 4 * 4

        self.fc1 = nn.Linear(flattened_dim, 1024)
        self.fc2 = nn.Linear(1024, output_dim)

    def forward(self, x):
        # x: [B, 1, img_size, img_size]
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(
            x)
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)  # Flatten to [B, 4096]
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # [B, output_dim]
        return x


class NeuroSymbolicPathfinderModel(nn.Module):
    def __init__(self, img_size, output_dim, provenance, k):
        super(NeuroSymbolicPathfinderModel, self).__init__()
        self.neural_net = NeuralNet(32, 256)

        # Scallop context
        self.scallop_file = "find_path.scl"
        self.ctx = scallopy.ScallopContext(provenance=provenance, k=k)
        self.ctx.import_file(self.scallop_file)
        self.ctx.set_input_mapping("dot", [(i,) for i in range(16)])
        self.ctx.set_input_mapping("dash", [
            (i, j) for i in range(1, 16) for j in range(16) if i != j
        ])

        self.eval_formula = self.ctx.forward_function("connected", jit=False,
                                                      recompile=False)

    def forward(self, images):
        out = self.neural_net(images)
        B = out.shape[0]

        dot_list = []
        dash_list = []

        # Here, we simplify the image into a 4 * 4 grid and
        # define a dot by having a dot in a cell of the grid,
        # or a dash by having connectivity between two cells
        dash_mapping = [(i, j) for i in range(1, 16) for j in range(16) if
                        i != j]

        for b in range(B):
            sample = out[b].detach().cpu()
            # dot: 0~15
            sample_dot = [(sample[i], (i,)) for i in range(16)]
            # dash: 16~255
            sample_dash = [(sample[16 + idx], dash_mapping[idx]) for idx in
                           range(len(dash_mapping))]
            dot_list.append(sample_dot)
            dash_list.append(sample_dash)

        (mapping, probs) = self.eval_formula(dash=dash_list, dot=dot_list)

        return mapping, probs