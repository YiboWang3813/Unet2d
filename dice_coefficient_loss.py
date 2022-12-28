import torch
import torch.nn.functional as F
from torch import Tensor, nn


def build_target(target, num_classes=2, ignore_index=100): 
    # type: (Tensor, int, int) -> Tensor 
    """ 创建每个类别的Ground Truth """
    dice_target = target.clone() 
    if ignore_index >= 0: 
        ignore_mask = torch.eq(target, ignore_index) 
        dice_target[ignore_mask] = 0 
        # [N, H, W] -> [N, H, W, C] 
        dice_target = nn.functional.one_hot(dice_target, num_classes).float() 
        dice_target[ignore_mask] = ignore_index 
    else: 
        dice_target = nn.functional.one_hot(dice_target, num_classes).float() 

    return dice_target.permute(0, 3, 1, 2) 


def dice_coeff(x, target, ignore_index=100, epsilon=1e-6): 
    # type: (Tensor, Tensor, int, float) -> float 
    """ 计算一个batch中所有图像某个类别的dice coefficient """ 
    d = 0. 
    batch_size = x.shape[0] 
    for i in range(batch_size): 
        x_i = x[i].reshape(-1) 
        t_i = target[i].reshape(-1) 
        if ignore_index >= 0: 
            roi_mask = torch.ne(t_i, ignore_index) 
            x_i = x_i[roi_mask] 
            t_i = t_i[roi_mask] 

        top = torch.dot(x_i, t_i) 
        bottom = torch.sum(x_i) + torch.sum(t_i) 
        if bottom == 0: 
            bottom = 2 * top 

        d += (2 * top + epsilon) / (bottom + epsilon) 

    return d / batch_size 


def multiclass_dice_coeff(x, target, ignore_index=100, epsilon=1e-6): 
    # type: (Tensor, Tensor, int, float) -> float 
    """ 计算多类别的Dice系数 """ 
    dice = 0. 
    for channel in range(x.shape[1]): 
        dice += dice_coeff(x[:, channel, ...], target[:, channel, ...], ignore_index, epsilon) 

    return dice / x.shape[1] 


def dice_loss(x, target, multiclass=False, ignore_index=-100): 
    # type: (Tensor, Tensor, bool, int) -> float 
    """ 计算dice损失 """ 
    x = torch.softmax(x, dim=1) 
    fn = multiclass_dice_coeff if multiclass else dice_coeff 
    return 1 - fn(x, target, ignore_index) 
