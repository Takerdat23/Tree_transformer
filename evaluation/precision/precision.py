from sklearn.metrics import precision_score

class Precision:
    def compute_score(self, gts, res):
        if len(gts) > len(res):
            delta = len(gts) - len(res)
            res += ["<pad>"]*delta
        else:
            res = res[:len(gts)]

        precision = precision_score(gts, res, average="macro")
        return precision

    def __str__(self) -> str:
        return "Precision"