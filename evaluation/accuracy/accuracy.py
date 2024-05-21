from sklearn.metrics import accuracy_score

class Accuracy:
    def compute_score(self, gts, res):
        if len(gts) > len(res):
            delta = len(gts) - len(res)
            res += ["<pad>"]*delta
        else:
            res = res[:len(gts)]

        return accuracy_score(gts, res)

    def __str__(self) -> str:
        return "Accuracy"
