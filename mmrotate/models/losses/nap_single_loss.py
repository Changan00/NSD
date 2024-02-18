import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.losses.utils import weighted_loss
from ..builder import ROTATED_LOSSES


def _get_gt_mask(logits, target, use_sigmoid):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits)
    if use_sigmoid:
        if logits.size(1) == 1:
            t = torch.zeros_like(logits)
            mask = torch.cat((mask,t),1)
        else:
            t = torch.zeros_like(logits[:,:2])
            mask = torch.cat((mask,t),1)[:,:-1]
    mask = mask.scatter_(1, target.unsqueeze(1), 1)
    mask = mask.bool()
    return mask


def _get_other_mask(logits, target, use_sigmoid):
    target = target.reshape(-1)
    mask = torch.ones_like(logits)
    if use_sigmoid:
        if logits.size(1) == 1:
            t = torch.ones_like(logits)
            mask = torch.cat((mask,t),1)
        else:
            t = torch.ones_like(logits[:,:2])
            mask = torch.cat((mask,t),1)[:,:-1]
    mask = mask.scatter_(1, target.unsqueeze(1), 0)
    mask = mask.bool()
    return mask


def cat_mask(t, mask1, mask2, use_sigmoid, use_back, use_sample):
    if use_sigmoid==True or use_back==False:
        t1 = (t * mask1[:,:-1])[mask1[:,-1]!=1].sum(dim=1, keepdims=True)
        t2 = (t * mask2[:,:-1])[mask2[:,-1]!=0].sum(dim=1, keepdims=True)
    else:
        if use_sample == 'pos':
            t1 = (t * mask1)[mask1[:,-1]!=1].sum(dim=1, keepdims=True) 
            t2 = (t * mask2)[mask1[:,-1]!=1].sum(dim=1, keepdims=True)
        else:
            t1 = (t * mask1).sum(dim=1, keepdims=True) 
            t2 = (t * mask2).sum(dim=1, keepdims=True)

    rt = torch.cat([t1, t2], dim=1)
    return rt

@weighted_loss
def dkd_loss(pred, target, logits_teacher, label_weight,  
             alpha, beta, temperature, use_sigmoid, use_sample, use_back):
    logits_student = pred
    gt_mask = _get_gt_mask(logits_student, target, use_sigmoid)
    other_mask = _get_other_mask(logits_student, target, use_sigmoid)
    if use_sigmoid==True or use_back==True:
        pred_student = F.softmax(logits_student / temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    else:
        pred_student = F.softmax(logits_student[:, :-1] / temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher[:, :-1] / temperature, dim=1)
    # print("pred_student: ", pred_student)
    # print("pred_teacher: ", pred_teacher)
    # print("target: ", target)
    pred_student = cat_mask(pred_student, gt_mask, other_mask, use_sigmoid, use_back, use_sample)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask, use_sigmoid, use_back, use_sample)
    log_pred_student = torch.log(pred_student)

    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, reduction='sum')
        * (temperature**2)
        / target.shape[0]
    )
    # print("tckd_loss: ",tckd_loss)
    # print("tckd_loss",tckd_loss)
    if use_sigmoid:
        gt_mask = gt_mask[:,:-1]
    if use_sample == 'pos_neg_ignore':
        pred_teacher_part2 = F.softmax(
            (logits_teacher / temperature - 1000.0 * gt_mask), dim=1
        )
        log_pred_student_part2 = F.log_softmax(
            (logits_student / temperature - 1000.0 * gt_mask), dim=1
        )
    elif use_sample == 'pos_neg':
        if use_back:
            pred_teacher_part2 = F.softmax(
                (logits_teacher / temperature - 1000.0 * gt_mask)[label_weight==1], dim=1
            )
            log_pred_student_part2 = F.log_softmax(
                (logits_student / temperature - 1000.0 * gt_mask)[label_weight==1], dim=1
            )
        else:
            pred_teacher_part2 = F.softmax(
                (logits_teacher[:,:-1] / temperature - 1000.0 * gt_mask[:,:-1])[label_weight==1], dim=1
            )
            log_pred_student_part2 = F.log_softmax(
                (logits_student[:,:-1] / temperature - 1000.0 * gt_mask[:,:-1])[label_weight==1], dim=1
            )
    elif use_sample == "pos":
        if use_back:
            pred_teacher_part2 = F.softmax(
                (logits_teacher / temperature)[other_mask[:,-1]==1], dim=1
            )
            log_pred_student_part2 = F.log_softmax(
                (logits_student / temperature)[other_mask[:,-1]==1], dim=1
            )
        else:
            pred_teacher_part2 = F.softmax(
                (logits_teacher[:,:-1] / temperature - 1000.0 * gt_mask[:,:-1])[other_mask[:,-1]==1], dim=1
            )
            log_pred_student_part2 = F.log_softmax(
                (logits_student[:,:-1] / temperature - 1000.0 * gt_mask[:,:-1])[other_mask[:,-1]==1], dim=1
            )
    elif use_sample == "neg":
        if use_back:
            pred_teacher_part2 = F.softmax(
                (logits_teacher / temperature - 1000.0 * gt_mask)[other_mask[:,-1]==0], dim=1
            )
            log_pred_student_part2 = F.log_softmax(
                (logits_student / temperature - 1000.0 * gt_mask)[other_mask[:,-1]==0], dim=1
            )
        else:
            pred_teacher_part2 = F.softmax(
                (logits_teacher[:,:-1] / temperature - 1000.0 * gt_mask[:,:-1])[other_mask[:,-1]==0], dim=1
            )
            log_pred_student_part2 = F.log_softmax(
                (logits_student[:,:-1] / temperature - 1000.0 * gt_mask[:,:-1])[other_mask[:,-1]==0], dim=1
            )
    elif use_sample == "pos_negn":
        if use_back:
            pred_teacher_part2 = torch.zeros_like(logits_teacher)
            pred_teacher_part2[other_mask[:,-1]==0] = F.softmax(
                (logits_teacher / temperature - 1000.0 * gt_mask)[other_mask[:,-1]==0], dim=1
            )
            pred_teacher_part2[other_mask[:,-1]==1] = F.softmax(
                (logits_teacher / temperature)[other_mask[:,-1]==1], dim=1
            )
            log_pred_student_part2 = torch.zeros_like(logits_teacher)
            log_pred_student_part2[other_mask[:,-1]==0] = F.log_softmax(
                (logits_student / temperature - 1000.0 * gt_mask)[other_mask[:,-1]==0], dim=1
            )
            log_pred_student_part2[other_mask[:,-1]==1] = F.log_softmax(
                (logits_student / temperature)[other_mask[:,-1]==1], dim=1
            )
        else:
            pred_teacher_part2 = torch.zeros_like(logits_teacher[:,:-1])
            pred_teacher_part2[other_mask[:,-1]==0] = F.softmax(
                (logits_teacher[:,:-1] / temperature - 1000.0 * gt_mask[:,:-1])[other_mask[:,-1]==0], dim=1
            )
            pred_teacher_part2[other_mask[:,-1]==1] = F.softmax(
                (logits_teacher[:,:-1] / temperature)[other_mask[:,-1]==1], dim=1
            )
            log_pred_student_part2 = torch.zeros_like(logits_teacher[:,:-1])
            log_pred_student_part2[other_mask[:,-1]==0] = F.log_softmax(
                (logits_student[:,:-1] / temperature - 1000.0 * gt_mask[:,:-1])[other_mask[:,-1]==0], dim=1
            )
            log_pred_student_part2[other_mask[:,-1]==1] = F.log_softmax(
                (logits_student[:,:-1] / temperature)[other_mask[:,-1]==1], dim=1
            )
    elif use_sample == "negn":
        if use_back:
            pred_teacher_part2 = torch.zeros_like(logits_teacher)
            pred_teacher_part2[other_mask[:,-1]==0] = F.softmax(
                (logits_teacher / temperature - 1000.0 * gt_mask)[other_mask[:,-1]==0], dim=1
            )
            log_pred_student_part2 = torch.zeros_like(logits_teacher)
            log_pred_student_part2[other_mask[:,-1]==0] = F.log_softmax(
                (logits_student / temperature - 1000.0 * gt_mask)[other_mask[:,-1]==0], dim=1
            )
            
        else:
            pred_teacher_part2 = torch.zeros_like(logits_teacher[:,:-1])
            pred_teacher_part2[other_mask[:,-1]==0] = F.softmax(
                (logits_teacher[:,:-1] / temperature - 1000.0 * gt_mask[:,:-1])[other_mask[:,-1]==0], dim=1
            )
            log_pred_student_part2 = torch.zeros_like(logits_teacher[:,:-1])
            log_pred_student_part2[other_mask[:,-1]==0] = F.log_softmax(
                (logits_student[:,:-1] / temperature - 1000.0 * gt_mask[:,:-1])[other_mask[:,-1]==0], dim=1
            )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='sum')
        * (temperature**2)
        / target.shape[0]
    )
    # print("nckd_loss: ",nckd_loss)

    # return alpha * tckd_loss + beta * nckd_loss
    return beta * nckd_loss


@ROTATED_LOSSES.register_module()
class NAPSingleLoss(nn.Module):
    """Decoupled Knowledge Distillation(CVPR 2022)"""

    def __init__(self, alpha, beta, T, warmup, loss_weight=1, use_sigmoid=True, use_sample='pos', use_back=True):
        super(NAPSingleLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = T
        self.warmup = warmup
        self.loss_weight = loss_weight
        self.use_sigmoid = use_sigmoid
        self.use_sample = use_sample
        self.use_back = use_back

    def forward(self, logits_student, target, soft_labels, label_weights, epoch):
        # loss_dkd = self.loss_weight * dkd_loss(
        loss_dkd = self.loss_weight * min(epoch / self.warmup, 1.0) * dkd_loss(
            logits_student,
            target,
            logits_teacher=soft_labels,
            label_weight=label_weights,
            alpha=self.alpha,
            beta=self.beta,
            temperature=self.temperature,
            use_sigmoid=self.use_sigmoid,
            use_sample = self.use_sample,
            use_back=self.use_back
        )
        return loss_dkd
