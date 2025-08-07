import torch
from torch import Tensor


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)

# def dice_coeff( input: Tensor, target: Tensor, weight: Tensor = None, reduce_batch_first: bool = False, epsilon: float = 1e-6) -> Tensor:
#     assert input.size() == target.size()
#     assert input.dim() == 3 or not reduce_batch_first
#
#     sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)
#
#     inter = 2 * (input * target).sum(dim=sum_dim)
#     sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
#     sets_sum = torch.where(sets_sum == 0, inter, sets_sum)
#
#     dice = (inter + epsilon) / (sets_sum + epsilon)  # shape=(N,)
#
#     if weight is not None:
#         weight = weight.to(dice.device)
#         dice = dice * weight
#         return dice.sum() / (weight.sum() + 1e-12)
#     else:
#         return dice.mean()
#
#
# def multiclass_dice_coeff(input: Tensor, target: Tensor, class_weight: Tensor = None, epsilon: float = 1e-6) -> Tensor:
#     B, C, H, W = input.shape
#     # First, merge the batch and class dimensions: N = B * C
#     inp = input.flatten(0, 1)        # (B*C, H, W)
#     tgt = target.flatten(0, 1)       # (B*C, H, W)
#
#     # If `class_weight` is passed, expand it to each of the (B*C) terms:
#     if class_weight is not None:
#         # repeat each class weight B
#         w = class_weight.repeat_interleave(B)
#     else:
#         w = None
#
#     return dice_coeff(inp, tgt, weight=w, reduce_batch_first=True, epsilon=epsilon)
#
#
# def dice_loss(
#     input: Tensor,
#     target: Tensor,
#     multiclass: bool = False,
#     class_weight: Tensor = None
# ) -> Tensor:
#     """
#     multiclass=True :
#       input (B,C,H,W)，target (B,C,H,W)
#     multiclass=False:
#       input (B,H,W) or (H,W), target
#     """
#     if multiclass:
#         return 1 - multiclass_dice_coeff(input, target, class_weight)
#     else:
#         return 1 - dice_coeff(input, target, weight=class_weight, reduce_batch_first=True)