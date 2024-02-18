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
from tools.analysis_tools.similarity_visualization import draw_similarity_map
from tools.feature_visualization import draw_feature_map

@ROTATED_DETECTORS.register_module()
class KDRotatedSingleStageDetector(RotatedSingleStageDetector):
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
                 fpnkd_cfg,
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
        self.fpnkd_cfg = fpnkd_cfg
        # Build teacher model
        if isinstance(teacher_config, str):
            teacher_config = mmcv.Config.fromfile(teacher_config)
        self.teacher_model = build_detector(teacher_config['model'])
        if teacher_ckpt is not None:
            load_checkpoint(
                self.teacher_model, teacher_ckpt, map_location='cpu')
        if 'neck-adapt' in fpnkd_cfg.type:
            self.neck_adapt = []
            in_channels = fpnkd_cfg.neck_in_channels
            out_channels = fpnkd_cfg.neck_out_channels
            for i in range(len(in_channels)):
                self.neck_adapt.append(
                    torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels[i], out_channels[i], kernel_size=3, padding=1),
                        # nn.ReLU(),
                        torch.nn.Sequential(),
                    )
                )
            self.neck_adapt = torch.nn.ModuleList(self.neck_adapt)
            for i in range(len(self.neck_adapt)):
                self.neck_adapt[i][0].weight.data.normal_().fmod_(2).mul_(0.0001).add_(0)
                self.neck_adapt[i].cuda()
    def set_epoch(self, epoch): 
        self.bbox_head.epoch = epoch 

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
        #super(KnowledgeDistillationRotatedSingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        # print("img: ", img)
        with torch.no_grad():
            print(next(self.teacher_model.parameters()).device)
            teacher_x = self.teacher_model.extract_feat(img)
            # print("teacher_x: ", teacher_x)
            out_teacher = self.teacher_model.bbox_head(teacher_x)
        
        if not self.output_feature:
            losses, bbox_weights = self.bbox_head.forward_train(x, out_teacher, img_metas,
                                                  gt_bboxes, gt_labels,
                                                  gt_bboxes_ignore)
        else:
            losses, bbox_weights = self.bbox_head.forward_train(x, out_teacher, teacher_x,
                                                  img_metas, gt_bboxes,
                                                  gt_labels, gt_bboxes_ignore, **kwargs)
        # bbox_weights:shape([5,2,num_anchors,5])其中第二个5为边界框5个参数
        if self.fpnkd_cfg.feature_kd == True:
            kd_decay = 1 - self.bbox_head.epoch / 12
            featmap_sizes = [featmap.size()[-2:] for featmap in x]
            neck_mask_batch = bbox_weights
            neck_feat = x
            neck_feat_t = teacher_x
            losskd_neck = torch.Tensor([0]).cuda()
            losskd_neck_ = torch.Tensor([0]).cuda()
            losskd_neck_back = torch.Tensor([0]).cuda()
            losskd_neck_back_ = torch.Tensor([0]).cuda()
            for i, _neck_feat in enumerate(neck_feat):
                mask_hint = neck_mask_batch[i][:,:,0].reshape(2, featmap_sizes[i][0], featmap_sizes[i][1], -1)
                mask_hint = mask_hint.unsqueeze(1).repeat(1, _neck_feat.size(1), 1, 1, 1)
                norms = max(1.0, mask_hint.sum() * 2)
                if 'neck-adapt' in self.fpnkd_cfg.type:
                    neck_feat_adapt = self.neck_adapt[i](_neck_feat)
                else:
                    neck_feat_adapt = _neck_feat
    
                if 'pixel-wise' in self.fpnkd_cfg.type:               
                    if 'L1' in self.fpnkd_cfg.type:
                        diff = torch.abs(neck_feat_adapt - neck_feat_t[i])
                        loss = torch.where(diff < 1.0, diff, diff**2)
                        losskd_neck += (loss * mask_hint).sum() / norms
                    elif 'Div' in self.fpnkd_cfg.type:
                        losskd_neck += (torch.pow(1 - neck_feat_adapt / (neck_feat_t[i] + 1e-8), 2) * mask_hint).sum() / norms
                    elif 'neck-decouple' in self.fpnkd_cfg.type:
                        for j in range(mask_hint.size(-1)):
                            norms_back = max(1.0, (1 - mask_hint).sum() * 2)
                            losskd_neck_back_ += (torch.pow(neck_feat_adapt - neck_feat_t[i], 2) * 
                                            (1 - mask_hint[:,:,:,:,j])).sum() / norms_back
                            losskd_neck_ += (torch.pow(neck_feat_adapt - neck_feat_t[i], 2) * mask_hint[:,:,:,:,j]).sum() / norms
                        losskd_neck_back += losskd_neck_back_/mask_hint.size(-1)
                        losskd_neck += losskd_neck_/mask_hint.size(-1)
                    else:
                        losskd_neck = losskd_neck + (torch.pow(neck_feat_adapt - neck_feat_t[i], 2) * 
                                                    mask_hint).sum() / norms

                if 'pixel-wise' in self.fpnkd_cfg.type:
                    losskd_neck = losskd_neck / len(neck_feat)
                    losskd_neck = losskd_neck * self.fpnkd_cfg.hint_neck_w
                    if 'decay' in self.fpnkd_cfg.type:
                        losskd_neck *= kd_decay
                    losses['losskd_neck'] = losskd_neck

                if 'neck-decouple' in self.fpnkd_cfg.type:
                    losskd_neck_back = losskd_neck_back / len(neck_feat)
                    losskd_neck_back = losskd_neck_back * self.fpnkd_cfg.hint_neck_back_w
                    if 'decay' in self.fpnkd_cfg.type:
                        losskd_neck_back *= kd_decay
                    losses['losskd_neck_back'] = losskd_neck_back
            # loss, log_vars = parse_losses(losses)
            # outputs = dict(
            #     loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))
        return losses

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

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        x = self.extract_feat(img)
        device = img.device
        self.teacher_model.to(device)
        tea_x = self.teacher_model.extract_feat(img)
        x_s = []
        for i, _neck_feat in enumerate(x):
            neck_feat_adapt = self.neck_adapt[i](_neck_feat)
            x_s.append(neck_feat_adapt)
        # draw_similarity_map(x, tea_x, img_metas, 25, 
        #                     "/root/mmrotate/train_nsd/FIGS/similarity_fig/12/{}".format(img_metas[0]['ori_filename'][:-4]), img_metas[0]['ori_filename'][:-4])    
        # draw_feature_map(x, img_metas, "/root/mmrotate/train_nsd/FIGS/similarity_fig/20/{}".format(img_metas[0]['ori_filename'][:-4]), img_metas[0]['ori_filename'][:-4])
        outs = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        bbox_results = [
            rbbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results

    def show_result(self,
                    img,
                    result,
                    score_thr=0.3,
                    bbox_color=(183, 58, 103),
                    text_color='green',
                    thickness=2,
                    font_scale=0.5,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
            text_color (str or tuple or :obj:`Color`): Color of texts.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (torch.Tensor): Only if not `show` or `out_file`
        """
        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        # draw segmentation masks
        if segm_result is not None and len(labels) > 0:
            segms = mmcv.concat_list(segm_result)
            inds = np.where(bboxes[:, -1] > score_thr)[0]
            np.random.seed(42)
            color_masks = [
                np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                for _ in range(max(labels) + 1)
            ]
            for i in inds:
                i = int(i)
                color_mask = color_masks[labels[i]]
                mask = segms[i].astype(bool)
                img[mask] = img[mask] * 0.5 + color_mask * 0.5
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
        # draw bounding rboxes
        imshow_det_rbboxes(
            img,
            bboxes,
            labels,
            class_names=self.CLASSES,
            score_thr=score_thr,
            bbox_color=bbox_color,
            text_color=text_color,
            thickness=thickness,
            font_scale=font_scale,
            win_name=win_name,
            show=show,
            wait_time=wait_time,
            out_file=out_file)

        if not (show or out_file):
            return img
    
