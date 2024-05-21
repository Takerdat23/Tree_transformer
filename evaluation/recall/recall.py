from sklearn.metrics import recall_score

class Recall:
    def compute_score(self, gts, res):
        if len(gts) > len(res):
            delta = len(gts) - len(res)
            res += ["<pad>"]*delta
        else:
            res = res[:len(gts)]

        recall = recall_score(gts, res, average="macro")
        return recall

    def __str__(self) -> str:
        return "Recall"