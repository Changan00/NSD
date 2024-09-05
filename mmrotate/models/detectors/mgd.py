# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
from numpy.lib.twodim_base import tri
import torch
from mmcv.runner import load_checkpoint

from .. import build_detector
from ..builder import ROTATED_DETECTORS
from mmrotate.core import imshow_det_rbboxes
from .single_stage import RotatedSingleStageDetector
from mmrotate.core import rbbox2result

@ROTATED_DETECTORS.register_module()
class MGD(RotatedSingleStageDetector):
    r"""Implementation of `Distilling the Knowledge in a Neural Network.
    <https://arxiv.org/abs/1503.02531>`_.
    Args:
        teacher_config (str | dict): Config file path
            or the config object of teacher model.
        teacher_ckpt (str, optional): Checkpoint path of teacher model.
            If left as None, the model will not load any weights.
    """

    def __init__(self,
                 backbone,
                 neck, 
                 bbox_head,
                 teacher_config,
                 alpha_mgd=0.00002,
                 lambda_mgd=0.65,
                 output_feature=False,
                 teacher_ckpt=None,
                 eval_teacher=True,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(backbone, neck, bbox_head, train_cfg, test_cfg,
                         pretrained,init_cfg)
        self.eval_teacher = eval_teacher
        self.output_feature = output_feature
        self.alpha_mgd = alpha_mgd
        self.lambda_mgd = lambda_mgd

        # Build teacher model
        if isinstance(teacher_config, str):
            teacher_config = mmcv.Config.fromfile(teacher_config)
        self.teacher_model = build_detector(teacher_config['model'])
        if teacher_ckpt is not None:
            load_checkpoint(
                self.teacher_model, teacher_ckpt, map_location='cpu')
        
        self.generation = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True), 
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1))

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None, **kwargs):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        self.teacher_model.eval()
        with torch.no_grad():
            tea_feats = self.teacher_model.extract_feat(img)
            tea_outs = self.teacher_model.bbox_head(tea_feats)
        stu_feats = self.extract_feat(img)
        stu_outs = self.bbox_head(stu_feats)
        loss_inputs = stu_outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.bbox_head.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        loss_mgd = self.get_dis_loss(stu_feats, tea_feats)*self.alpha_mgd
        losses.update(dict(loss_mgd=loss_mgd))
        return losses

    def get_dis_loss(self, stu_feats, tea_feats):
        dis_loss = 0.0
        for i, (preds_S, preds_T) in enumerate(zip(stu_feats, tea_feats)):
            loss_mse = torch.nn.MSELoss(reduction='sum')
            N, C, H, W = preds_T.shape

            device = preds_S.device
            mat = torch.rand((N,1,H,W)).to(device)
            mat = torch.where(mat>1-self.lambda_mgd, 0, 1).to(device)

            masked_fea = torch.mul(preds_S, mat)
            new_fea = self.generation(masked_fea)

            dis_loss += loss_mse(new_fea, preds_T)/N

        return dis_loss

    def cuda(self, device=None):
        """Since teacher_model is registered as a plain object, it is necessary
        to put the teacher model to cuda when calling cuda function."""
        self.teacher_model.cuda(device=device)
        return super().cuda(device=device)

    def train(self, mode=True):
        """Set the same train mode for teacher and student model."""
        if self.eval_teacher:
            self.teacher_model.train(False)
        else:
            self.teacher_model.train(mode)
        super().train(mode)

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

    