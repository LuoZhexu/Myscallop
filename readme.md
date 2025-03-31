# MyScallop: Custom Experiments and Implementations

## 1. `/examples/myscal`

This directory includes independent experiments and reproductions of existing tasks.

### Custom Scallop file:

- **`genetype.scl`**
   Implements a Scallop logic program that calculates the probability of descendants inheriting a genetic disease given the genotypes of their ancestors.
- **`pacman_find_path.py`**
   A Python program that randomly generates grids with varying enemy counts and positions, as well as random start and end points. It uses Scallop logic to determine:
  - Whether Pacman can reach the target location
  - The shortest path if reachable

### Reproduction of the Handwritten Formula (HWF) Task:

All other files in this directory (`train.py` and associated utilities) reproduce the HWF task described in the original Scallop paper. This implementation was done without referencing the original Scallop source code.

- **Dataset:** [Handwritten Formula Dataset](https://drive.google.com/file/d/1VW--BO_CSxzB9C7-ZpE3_hrZbXDqlMU-/view)

- After setting up the dataset as described, run the experiment by executing:

  ```bash
  python train.py
  ```

## 2. `/experiments/pathfinder`

This directory contains my custom implementation of the Pathfinder task using Scallop.

### Task Description:

- Images are simplified into a 4x4 grid.
- Dots represent cell activations, and dashes represent connections between adjacent cells.
- A neural network predicts the existence probabilities of dots and dashes.
- Scallop logic is used afterward to determine connectivity between specified points.

### Dataset:

- **Source:** [Pathfinder32 Dataset (Kaggle)](https://www.kaggle.com/datasets/hajarbel04/pathfinder32/discussion?sort=hotness)

### Running the Experiment:

After properly configuring the dataset as instructed, execute:

```bash
python train.py
```

