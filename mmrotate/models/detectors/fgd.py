import torch.nn as nn
import torch.nn.functional as F
import torch
import mmcv
from mmrotate.models.detectors.base import BaseDetector
from mmrotate.models import build_detector
from mmrotate.core import norm_angle, obb2poly_np, poly2obb_np
from mmcv.runner import  load_checkpoint, _load_checkpoint, load_state_dict
from ..builder import ROTATED_DETECTORS
from collections import OrderedDict
from .single_stage import RotatedSingleStageDetector

import torch
import torch.nn as nn
import torch.nn.functional as F
import mmcv
from mmcv.runner import load_checkpoint

from mmcv.cnn import constant_init, kaiming_init

import numpy as np
import cv2
@ROTATED_DETECTORS.register_module()
class FGD(RotatedSingleStageDetector):
    """
    cvpr22 feature KD based on focal and global.

    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,

                 teacher_config,
                 teacher_ckpt=None,

                 # idea
                 focal_kd=True,
                 global_kd=True,
                 temp=0.5,
                 alpha_fgd=0.001,
                 beta_fgd=0.0005,
                 gamma_fgd=0.001,
                 lambda_fgd=0.000005,

                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(FGD, self).__init__(backbone, neck, bbox_head, train_cfg,
                                  test_cfg, pretrained, init_cfg)
        # Build teacher model
        self.eval_teacher = True
        if isinstance(teacher_config, str):
            teacher_config = mmcv.Config.fromfile(teacher_config)
        self.teacher_model = build_detector(teacher_config['model'])
        if teacher_ckpt is not None:
            load_checkpoint(
                self.teacher_model, teacher_ckpt, map_location='cpu')
        self.freeze_models()
            
        # idea
        self.focal_kd = focal_kd,
        self.global_kd = global_kd,
        self.temp = temp
        self.alpha_fgd = alpha_fgd
        self.beta_fgd = beta_fgd
        self.gamma_fgd = gamma_fgd
        self.lambda_fgd = lambda_fgd

        self.conv_mask_s = nn.Conv2d(256, 1, kernel_size=1)
        self.conv_mask_t = nn.Conv2d(256, 1, kernel_size=1)
        self.channel_add_conv_s = nn.Sequential(
            nn.Conv2d(256, 256 // 2, kernel_size=1),
            nn.LayerNorm([256 // 2, 1, 1]),
            nn.ReLU(inplace=True),  # yapf: disable
            nn.Conv2d(256 // 2, 256, kernel_size=1))
        self.channel_add_conv_t = nn.Sequential(
            nn.Conv2d(256, 256 // 2, kernel_size=1),
            nn.LayerNorm([256 // 2, 1, 1]),
            nn.ReLU(inplace=True),  # yapf: disable
            nn.Conv2d(256 // 2, 256, kernel_size=1))

        self.reset_parameters() # ？？？

    def last_zero_init(self, m):
        if isinstance(m, nn.Sequential):
            constant_init(m[-1], val=0)
        else:
            constant_init(m, val=0)

    def reset_parameters(self):
        kaiming_init(self.conv_mask_s, mode='fan_in')
        kaiming_init(self.conv_mask_t, mode='fan_in')
        self.conv_mask_s.inited = True
        self.conv_mask_t.inited = True

        self.last_zero_init(self.channel_add_conv_s)
        self.last_zero_init(self.channel_add_conv_t)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):

        self.teacher_model.eval()
        with torch.no_grad():
            tea_feats = self.teacher_model.extract_feat(img)
            tea_outs = self.teacher_model.bbox_head(tea_feats)

        # 验证教师pth文件的权重是否被成功读入
        # for name, parameters in self.tea_model.named_parameters():
        #     if name == "neck.lateral_convs.1.conv.bias":
        #         print(parameters)

        stu_feats = self.extract_feat(img)
        stu_outs = self.bbox_head(stu_feats)
        loss_inputs = stu_outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.bbox_head.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        # idea
        loss_focal, loss_global = 0, 0

        for i, (stu_x,tea_x) in enumerate(zip(stu_feats, tea_feats)):
            N, C, H, W = stu_x.shape
            S_attention_t, C_attention_t = self.get_attention(tea_x, self.temp)
            S_attention_s, C_attention_s = self.get_attention(stu_x, self.temp)

            if self.focal_kd:
                Mask_fg = torch.zeros_like(S_attention_t)
                Mask_bg = torch.ones_like(S_attention_t)
                # wmin,wmax,hmin,hmax = [],[],[],[]
                for i in range(N):
                    new_boxxes = torch.ones_like(gt_bboxes[i])
                    new_boxxes[:, 0] = gt_bboxes[i][:, 0]/img_metas[i]['img_shape'][1]*W
                    new_boxxes[:, 2] = gt_bboxes[i][:, 2]/img_metas[i]['img_shape'][1]*W
                    new_boxxes[:, 1] = gt_bboxes[i][:, 1]/img_metas[i]['img_shape'][0]*H
                    new_boxxes[:, 3] = gt_bboxes[i][:, 3]/img_metas[i]['img_shape'][0]*H
                    new_boxxes[:, 4] = gt_bboxes[i][:, 4]

                    # wmin.append(torch.floor(new_boxxes[:, 0]).int())
                    # wmax.append(torch.ceil(new_boxxes[:, 2]).int())
                    # hmin.append(torch.floor(new_boxxes[:, 1]).int())
                    # hmax.append(torch.ceil(new_boxxes[:, 3]).int())

                    # area = 1.0/(torch.ceil(new_boxxes[:, 2]).int())/(torch.ceil(new_boxxes[:, 3]).int())
                    # area = area.unsqueeze(0)
                    # print(area.size())
                    # print(area)
                    polys = obb2poly_np(np.concatenate([new_boxxes.cpu().numpy(), torch.zeros_like(new_boxxes[:,:1]).cpu().numpy()], axis=1), 'le90')
                    # print(img.size())
                    init_img = img[i].permute(1,2,0)
                    init_img = init_img.cpu().numpy()
                    pad_img = mmcv.imresize(init_img, (W, H))
                    # print(pad_img)
                    for j in range(len(gt_bboxes[i])):
                        points = np.array([[polys[j][0],polys[j][1]],
                                        [polys[j][2],polys[j][3]],
                                        [polys[j][4],polys[j][5]],
                                        [polys[j][6],polys[j][7]]], np.int32)
                        p_img = cv2.fillPoly(pad_img, [points], color=(255, 255, 255))
                        area = torch.tensor(3/np.sum(p_img==255))
                        gt_w = (p_img==255)[:,:,0]
                        # print(gt_w.shape)
                        # print(Mask_fg.size())
                        Mask_fg[i, gt_w] = torch.maximum(Mask_fg[i, gt_w], area)

                    Mask_bg[i] = torch.where(Mask_fg[i]>0, 0, 1)
                    if torch.sum(Mask_bg[i]):
                        Mask_bg[i] /= torch.sum(Mask_bg[i])

                fg_loss, bg_loss = self.get_fea_loss(stu_x, tea_x, Mask_fg, Mask_bg,
                                   C_attention_s, C_attention_t, S_attention_s, S_attention_t)
                mask_loss = self.get_mask_loss(C_attention_s, C_attention_t, S_attention_s, S_attention_t)
                loss_focal += self.alpha_fgd * fg_loss + self.beta_fgd * bg_loss + self.gamma_fgd * mask_loss

            if self.global_kd:
                rela_loss = self.get_rela_loss(stu_x, tea_x)
                loss_global += self.lambda_fgd * rela_loss

        if self.focal_kd:
            losses.update(dict(loss_focal=loss_focal))
        if self.global_kd:
            losses.update(dict(loss_global=loss_global))

        return losses

    def get_attention(self, preds, temp):
        """ preds: Bs*C*W*H """
        N, C, H, W= preds.shape
        value = torch.abs(preds)
        # Bs*W*H
        fea_map = value.mean(axis=1, keepdim=True)
        S_attention = (H * W * F.softmax((fea_map/temp).view(N,-1), dim=1)).view(N, H, W)
        # Bs*C
        channel_map = value.mean(axis=2,keepdim=False).mean(axis=2,keepdim=False)
        C_attention = C * F.softmax(channel_map/temp, dim=1)
        return S_attention, C_attention

    def get_fea_loss(self, preds_S, preds_T, Mask_fg, Mask_bg, C_s, C_t, S_s, S_t):
        loss_mse = nn.MSELoss(reduction='sum')

        Mask_fg = Mask_fg.unsqueeze(dim=1)
        Mask_bg = Mask_bg.unsqueeze(dim=1)

        C_t = C_t.unsqueeze(dim=-1)
        C_t = C_t.unsqueeze(dim=-1)

        S_t = S_t.unsqueeze(dim=1)

        fea_t = torch.mul(preds_T, torch.sqrt(S_t))
        fea_t = torch.mul(fea_t, torch.sqrt(C_t))
        fg_fea_t = torch.mul(fea_t, torch.sqrt(Mask_fg))
        bg_fea_t = torch.mul(fea_t, torch.sqrt(Mask_bg))

        fea_s = torch.mul(preds_S, torch.sqrt(S_t))
        fea_s = torch.mul(fea_s, torch.sqrt(C_t))
        fg_fea_s = torch.mul(fea_s, torch.sqrt(Mask_fg))
        bg_fea_s = torch.mul(fea_s, torch.sqrt(Mask_bg))

        fg_loss = loss_mse(fg_fea_s, fg_fea_t) / len(Mask_fg)
        bg_loss = loss_mse(bg_fea_s, bg_fea_t) / len(Mask_bg)
        return fg_loss, bg_loss

    def get_mask_loss(self, C_s, C_t, S_s, S_t):
        mask_loss = torch.sum(torch.abs((C_s - C_t))) / len(C_s) + torch.sum(torch.abs((S_s - S_t))) / len(S_s)
        return mask_loss

    def spatial_pool(self, x, in_type):
        batch, channel, width, height = x.size()
        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        if in_type == 0:
            context_mask = self.conv_mask_s(x)
        else:
            context_mask = self.conv_mask_t(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = F.softmax(context_mask, dim=2)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(-1)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context

    def get_rela_loss(self, preds_S, preds_T):
        loss_mse = nn.MSELoss(reduction='sum')

        context_s = self.spatial_pool(preds_S, 0)
        context_t = self.spatial_pool(preds_T, 1)

        out_s = preds_S
        out_t = preds_T

        channel_add_s = self.channel_add_conv_s(context_s)
        out_s = out_s + channel_add_s

        channel_add_t = self.channel_add_conv_t(context_t)
        out_t = out_t + channel_add_t

        rela_loss = loss_mse(out_s, out_t) / len(out_s)

        return rela_loss

    def freeze_models(self):
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        """Set the same train mode for teacher and student model."""
        if self.eval_teacher:
            self.teacher_model.train(False)
        else:
            self.teacher_model.train(mode)
        super().train(mode)

    def cuda(self, device=None):
        """Since teacher_model is registered as a plain object, it is necessary
        to put the teacher model to cuda when calling cuda function."""
        self.teacher_model.cuda(device=device)
        return super().cuda(device=device)

    def __setattr__(self, name, value):
        """Set attribute, i.e. self.name = value
        This reloading prevent the teacher model from being registered as a
        nn.Module. The teacher module is registered as a plain object, so that
        the teacher parameters will not show up when calling
        ``self.parameters``, ``self.modules``, ``self.children`` methods.
        """
        if name == 'teacher_model':
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)


