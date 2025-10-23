import numpy as np
import math
from scipy import stats


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def cal_kappa(hist):
    if hist.sum() == 0:
        po = 0
        pe = 1
        kappa = 0
    else:
        po = np.diag(hist).sum() / hist.sum()
        pe = np.matmul(hist.sum(1), hist.sum(0).T) / hist.sum() ** 2
        if pe == 1:
            kappa = 0
        else:
            kappa = (po - pe) / (1 - pe)
    return kappa


class EvalMeter:
    def __init__(self, num_class: int):
        self.hist = np.zeros((num_class, num_class))
        self.num_class = num_class

    def reset(self):
        self.hist = np.zeros((self.num_class, self.num_class))

    def get_hist(self, image, label):
        hist = np.zeros((self.num_class, self.num_class))
        hist += fast_hist(image.flatten(), label.flatten(), self.num_class)
        return hist

    def update(self, preds, refs):
        for pred, ref in zip(preds, refs):
            infer_array = np.array(pred)
            label_array = np.array(ref)
            self.hist += self.get_hist(infer_array, label_array)

    def compute(self):
        hist_fg = self.hist[1:, 1:]
        c2hist = np.zeros((2, 2))
        c2hist[0][0] = self.hist[0][0]
        c2hist[0][1] = self.hist.sum(1)[0] - self.hist[0][0]
        c2hist[1][0] = self.hist.sum(0)[0] - self.hist[0][0]
        c2hist[1][1] = hist_fg.sum()
        hist_n0 = self.hist.copy()
        hist_n0[0][0] = 0
        kappa_n0 = cal_kappa(hist_n0)
        iu = np.diag(c2hist) / (c2hist.sum(1) + c2hist.sum(0) - np.diag(c2hist))
        IoU_fg = iu[1]
        IoU_mean = (iu[0] + iu[1]) / 2
        Sek = (kappa_n0 * math.exp(IoU_fg)) / math.e

        pixel_sum = self.hist.sum()
        change_pred_sum = pixel_sum - self.hist.sum(1)[0].sum()
        change_label_sum = pixel_sum - self.hist.sum(0)[0].sum()
        change_ratio = change_label_sum / pixel_sum
        SC_TP = np.diag(self.hist[1:, 1:]).sum()
        SC_Precision = SC_TP / change_pred_sum
        SC_Recall = SC_TP / change_label_sum
        Fscd = stats.hmean([SC_Precision, SC_Recall])

        return dict(
            kappa_n0=kappa_n0,
            f1=Fscd,
            miou=IoU_mean,
            sek=Sek
        )
