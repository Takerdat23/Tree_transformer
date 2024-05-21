from sklearn.metrics import f1_score

class F1:
    def compute_score(self, gts, res):
        if len(gts) > len(res):
            delta = len(gts) - len(res)
            res += ["<pad>"]*delta
        else:
            res = res[:len(gts)]

        return f1_score(gts, res, average="macro")

    def __str__(self) -> str:
        return "F1"
