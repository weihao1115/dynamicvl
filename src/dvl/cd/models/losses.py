import torch
import torch.nn.functional as F


@torch.jit.script
def select(y_pred: torch.Tensor, y_true: torch.Tensor, ignore_index: int):
    assert y_pred.ndim == 4 and y_true.ndim == 3
    c = y_pred.size(1)
    y_pred = y_pred.permute(0, 2, 3, 1).reshape(-1, c)
    y_true = y_true.reshape(-1)

    valid = y_true != ignore_index

    y_pred = y_pred[valid, :]
    y_true = y_true[valid]
    return y_pred, y_true


def dice_coeff(y_pred, y_true, weights: torch.Tensor, smooth_value: float = 1.0, sync_statistics=True):
    y_pred = y_pred[:, weights]
    y_true = y_true[:, weights]
    inter = torch.sum(y_pred * y_true, dim=0)
    z = y_pred.sum(dim=0) + y_true.sum(dim=0)
    z += smooth_value

    return ((2 * inter + smooth_value) / z).mean()


def dice_loss_with_logits(y_pred: torch.Tensor, y_true: torch.Tensor,
                          smooth_value: float = 1.0,
                          ignore_index: int = 255,
                          ignore_channel: int = -1,
                          *,
                          sync_statistics=True,
                          ):
    c = y_pred.size(1)
    y_pred, y_true = select(y_pred, y_true, ignore_index)
    weight = torch.as_tensor([True] * c, device=y_pred.device)
    if c == 1:
        y_prob = y_pred.sigmoid()
        return 1. - dice_coeff(y_prob, y_true.reshape(-1, 1), weight, smooth_value, sync_statistics)
    else:
        y_prob = y_pred.log_softmax(dim=1).exp()
        y_true = F.one_hot(y_true.long(), num_classes=c)
        if ignore_channel != -1:
            weight[ignore_channel] = False

        return 1. - dice_coeff(y_prob, y_true.type_as(y_pred), weight, smooth_value, sync_statistics)

