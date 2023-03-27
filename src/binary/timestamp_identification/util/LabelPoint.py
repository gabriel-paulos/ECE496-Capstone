from dataclasses import dataclass


@dataclass
class LabelPoint:
    token_index: int
    time_index: int
    score: float


@dataclass
class Segment:
    label: str
    score: float
    start: int
    end: int

    def __len__(self):
        return self.end - self.start
