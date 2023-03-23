import torch

from typing import Tuple, List
from util import LabelPoint


class Trellis:
    def __init__(self,
                 transcript: str,
                 labels: Tuple[str],
                 label_probabilities):

        self.path = None

        label_mapping = {label: index for index, label in enumerate(labels)}

        self.token_list = [label_mapping[c] for c in transcript]
        self.label_probabilities = label_probabilities

        num_frames = self.label_probabilities.size(0)
        num_tokens = len(self.token_list)

        self.graph = torch.empty((num_frames + 1, num_tokens + 1))
        self.graph[0, 0] = 0
        self.graph[1:, 0] = torch.cumsum(label_probabilities[:, 0], 0)
        self.graph[0, -num_tokens:] = float("-inf")
        self.graph[-num_tokens:, 0] = float("inf")

        for t in range(num_frames):
            self.graph[t + 1, 1:] = torch.maximum(
                self.graph[t, 1:] + self.label_probabilities[t, 0],
                self.graph[t, :-1] +
                self.label_probabilities[t, self.token_list])

    def backtrack(self) -> List[LabelPoint]:

        self.path = []

        token_index = self.graph.size(1) - 1
        time_index_start = torch.argmax(self.graph[:, token_index]).item()

        path = []
        for time_index in range(time_index_start, 0, -1):

            stay_score = self.graph[time_index - 1, token_index]
            + self.label_probabilities[time_index - 1, 0]

            change_score = self.graph[time_index - 1, token_index - 1]
            + self.label_probabilities[
                time_index - 1, self.token_list[token_index - 1]
            ]

            transition_probability = self.label_probabilities[
                time_index - 1, self.token_list[token_index - 1]
                if change_score > stay_score else 0].exp().item()

            self.path.append(LabelPoint(token_index - 1,
                                        time_index - 1,
                                        transition_probability))

            if change_score > stay_score:
                token_index -= 1
                if token_index == 0:
                    break
        else:
            raise ValueError("Alignment Failure")

        return path[::-1]
