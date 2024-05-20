import numpy as np
from sklearn.metrics import f1_score

class F1:
    def compute_score(self, gts, res):
        return f1_score(gts, res, average="macro")

    def __str__(self) -> str:
        return "F1"
