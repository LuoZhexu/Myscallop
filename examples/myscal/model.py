import torch
import torch.nn as nn
import torch.nn.functional as F
import scallopy

class SymbolCNN(nn.Module):
    def __init__(self, num_classes=14):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  # [B, 32, 45, 45]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 32, 22, 22]

            nn.Conv2d(32, 64, 3, padding=1),  # [B, 64, 22, 22]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 64, 11, 11]
        )

        self.mlp = nn.Sequential(
            nn.Linear(64 * 11 * 11, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):  # x: [B, 7, 1, 45, 45]
        B, S, C, H, W = x.shape
        x = x.view(B * S, C, H, W)  # flatten batch & seq
        x = self.cnn(x)  # [B*S, 64, 11, 11]
        x = x.view(B * S, -1)  # flatten
        x = self.mlp(x)  # [B*S, 14]
        x = x.view(B, S, -1)  # [B, 7, 14]
        return F.softmax(x, dim=-1)


class HWF(nn.Module):
    def __init__(self, k):
        super(HWF, self).__init__()

        self.symbol_cnn = SymbolCNN()
        self.scallop_file = "parse_formula.scl"
        self.symbols = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                "+", "-", "*", "/"]

        self.ctx = scallopy.ScallopContext(provenance="difftopbottomkclauses", k=3)
        self.ctx.import_file(self.scallop_file)
        self.ctx.set_non_probabilistic("length")
        self.ctx.set_input_mapping("symbol", [(i, s) for i in range(7) for s in
                                              self.symbols])
        self.computation = self.ctx.forward_function("result", jit=False, recompile=False)

    def forward(self, img_seq, img_seq_len):
        # Calculate the symbol distribution with shapes [B, 7, 14]
        symbols = self.symbol_cnn(img_seq).view(-1, 7, 14)
        B, _, _ = symbols.shape
        # Construct length information for each sample
        lengths = [[(l.item(),)] for l in img_seq_len]

        # Used to store the sampling results and dissections mapping for all tasks
        facts_per_task = []
        disj_mapping = []

        # Traverse each task (every sample in the batch)
        for task in range(B):
            task_facts = []
            task_disj = []
            # Traverse each symbol position in the current sample based on the actual length
            for pos in range(img_seq_len[task]):
                # Retrieve the probability distribution of the current location
                prob_dist = symbols[task, pos]
                cat_dist = torch.distributions.Categorical(prob_dist)
                sampled = cat_dist.sample((7,))
                unique_sampled = []
                for idx in sampled.tolist():
                    if idx not in unique_sampled:
                        unique_sampled.append(idx)
                # Build a fact list for the current symbol
                current_facts = [(prob_dist[i], (pos, self.symbols[i])) for i
                                 in unique_sampled]
                # Record the current position of facts
                start = len(task_facts)
                indices = list(range(start, start + len(current_facts)))
                task_disj.append(indices)
                # Append
                task_facts.extend(current_facts)
            facts_per_task.append(task_facts)
            disj_mapping.append(task_disj)

        mapping, probs = self.computation(symbol=facts_per_task,
                                          length=lengths,
                                          disjunctions={
                                              "symbol": disj_mapping})
        return ([r for (r,) in mapping], probs)


def run_scallop_on_sample(symbol_facts, length_val, ground_truth, program_file="parse_formula.scl"):
    ctx = scallopy.ScallopContext(provenance="difftopkproofs")
    ctx.import_file(program_file)

    ctx.add_facts("symbol", symbol_facts)
    ctx.add_facts("length", [(1.0, (length_val,))])
    ctx.run()

    result = ctx.relation("result")
    if not result:
        print("Warning: No result produced by Scallop")
        y_pred = torch.tensor(0.0, requires_grad=True)
        return (y_pred - ground_truth) ** 2

    max_p, max_val = max(result, key=lambda x: x[0].item())
    y_pred = torch.tensor(max_val[0], dtype=torch.float32, requires_grad=True)

    loss = (y_pred - ground_truth) ** 2
    return loss