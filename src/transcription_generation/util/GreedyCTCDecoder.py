import torch


class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank_index=0):
        super().__init__()
        self.labels = labels
        self.blank_index = blank_index

    def forward(self, label_probabilities: torch.Tensor) -> str:
        indices = torch.argmax(label_probabilities, dim=-1)
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank_index]
        return "".join([self.labels[i] for i in indices])
