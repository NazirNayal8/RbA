# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""
import logging

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from torchvision import transforms

from ..utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


def get_batch_avg(logits, label_ood):
    N, _, H, W = logits.shape
    m = logits.mean(1).mean()
    ma = - m.view(1,1,1).repeat(N,H,W) * label_ood
    return ma

class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        eos_coef,
        losses,
        num_points,
        oversample_ratio,
        importance_sample_ratio,
        smoothness_score,
        score_norm,
        outlier_loss_target,
        inlier_upper_threshold,
        outlier_lower_threshold,
        outlier_loss_func,
        pebal_reward, # pebal related
        ood_reg, # pebal related
        densehybrid_beta # densehybrid related
    ):
        """Create the criterion.
        Parameters:
        - num_classes:  number of object categories, omitting the special no-object category
        - matcher:      module able to compute a matching between targets and proposals
        - weight_dict:  dict containing as key the names of the losses and as values their relative weight.
        - eos_coef:     relative classification weight applied to the no-object category
        - losses:       list of all the losses to be applied. See get_loss for list of available losses.
        - [num_points, oversample_ratio, importance_sample_ratio]: PointRend Params
        - smoothness_score: score for which to apply local smoothness and sparsity losses
        - outlier_loss_target: the score function used in outlier loss with explicit outlier supervision
        - inlier_upper_threshold: the value below which the inlier pixels' target score should be pushed
                                , that is, the loss is higher if inlier score goes above this threshold
        - outlier_lower_threshold: the value above which the outlier pixels' target score should be pushed
                                , that is, the loss is higher if outlier score goes below this threshold

        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.smoothness_score = smoothness_score
        self.score_norm = score_norm
        self.outlier_loss_target = outlier_loss_target
        self.inlier_upper_threshold = inlier_upper_threshold
        self.outlier_lower_threshold = outlier_lower_threshold
        self.outlier_loss_func = outlier_loss_func
        self.pebal_reward = pebal_reward  # 4.5
        self.ood_reg = ood_reg  # 0.1
        self.densehybrid_beta = densehybrid_beta

        if smoothness_score not in ["none", "nls", "energy", "softmax_entropy"]:
            raise ValueError(
                f"Smoothness score should be one of [none, nls, energy, softmax_entropy], given was {smoothness_score}")

        if outlier_loss_target not in ["none", "nls", "energy", "softmax_entropy", "sum_entropy"]:
            raise ValueError(
                f"outlier_loss_target should be one of [none, nls, energy, softmax_entropy, sum_entropy], given was {smoothness_score}")

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J]
                                     for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(
            1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}
        return losses

    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]
        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def smoothness_loss(self, outputs, targets, indices, num_masks):
        """
        Smoothness regularization loss to encourage nearby pixels to have similar negative
        logit sum. 

        Adapted from https://github.com/tianyu0207/PEBAL/blob/main/code/losses.py

        TODO: in case of extra memory consumption, add POINT REND filtering
        """
        mask_logits = outputs["pred_masks"]
        class_logits = outputs["pred_logits"]
        class_logits = F.softmax(class_logits, dim=-1)[..., :-1]
        mask_logits = mask_logits.sigmoid()
        logits = torch.einsum("bqc,bqhw->bchw", class_logits, mask_logits)

        if self.smoothness_score == "nls":
            score = -logits.sum(dim=1)  # -> (B, H, W)
        elif self.smoothness_score == "energy":
            score = -torch.logsumexp(logits, dim=1)  # -> (B, H, W)
        elif self.outlier_loss_target == "softmax_entropy":
            score = torch.special.entr(logits.softmax(dim=1)).sum(dim=1)
        else:
            raise ValueError(
                f"Undefined Smoothness Score: f{self.smoothness_score}")

        score_h_shifted = torch.zeros_like(score)
        score_h_shifted[:, :-1, :] = score[:, 1:, :]
        score_h_shifted[:, -1, :] = score[:, -1, :]

        score_w_shifted = torch.zeros_like(score)
        score_w_shifted[:, :, :-1] = score[:, :, 1:]
        score_w_shifted[:, :, -1] = score[:, :, -1]

        loss = (torch.sum((score_h_shifted - score) ** 2) +
                torch.sum((score_w_shifted - score) ** 2)) / 2

        return {"smoothness_loss": loss}

    def sparsity_loss(self, outputs, targets, indices, num_masks):
        """
        Used to encourage sparsity of the negative logits sum score. 

        In the case of OOD supervision, targets will contain an entry 
        with key: "outlier_masks". In this case, sparsity will be computed 
        only for OOD regions, following the same method as the source below.

        Adapted from https://github.com/tianyu0207/PEBAL/blob/main/code/losses.py

        TODO: in case of extra memory consumption, add POINT REND filtering
        """
        mask_logits = outputs["pred_masks"]
        class_logits = outputs["pred_logits"]
        class_logits = F.softmax(class_logits, dim=-1)[..., :-1]
        mask_logits = mask_logits.sigmoid()
        logits = torch.einsum("bqc,bqhw->bchw", class_logits, mask_logits)

        if self.smoothness_score == "nls":
            score = -logits.sum(dim=1)  # -> (B, H, W)
        elif self.smoothness_score == "energy":
            score = -torch.logsumexp(logits, dim=1)  # -> (B, H, W)
        elif self.outlier_loss_target == "softmax_entropy":
            score = torch.special.entr(logits.softmax(dim=1)).sum(dim=1)
        else:
            raise ValueError(
                f"Undefined Smoothness Score: f{self.smoothness_loss}")

        if "outlier_masks" in targets[0]:
            outlier_masks = torch.cat([x["outlier_masks"].unsqueeze(
                0) for x in targets], dim=0)  # -> (B, H, W)
            ood_mask = (outlier_masks == 1)
            score = F.interpolate(score.unsqueeze(
                1), size=outlier_masks.shape[-2:], mode="bilinear", align_corners=True).squeeze(1)
            loss = torch.mean(torch.norm(score[ood_mask], dim=0))
        else:
            loss = torch.tensor(0., device=score.device)

        return {"sparsity_loss": loss}
    
    def gambler_loss(self, outputs, targets, indices, num_masks):

        mask_logits = outputs["pred_masks"]
        class_logits = outputs["pred_logits"]
        class_logits = F.softmax(class_logits, dim=-1)
        mask_logits = mask_logits.sigmoid()
        
        outlier_masks = torch.cat([x["outlier_masks"].unsqueeze(
            0) for x in targets], dim=0)  # -> (B, H, W)
        
        labels = torch.cat([x["sem_seg"].unsqueeze(
            0) for x in targets], dim=0)  # -> (B, H, W)

        logits = torch.einsum("bqc,bqhw->bchw", class_logits, mask_logits)
        logits = F.interpolate(logits, size=outlier_masks.shape[-2:], mode="bilinear", align_corners=True)

        probs = logits.softmax(dim=1)


        ood_mask = (outlier_masks == 1)
        id_mask = (outlier_masks == 0)

        true_pred, reservation = probs[:, :-1, :, :], probs[:, -1, :, :]
        
        reward = torch.logsumexp(logits[:, :-1, :, :], dim=1).pow(2)

        if reward.nelement() > 0:
            gaussian_smoothing = transforms.GaussianBlur(7, sigma=1)
            reward = reward.unsqueeze(0)
            reward = gaussian_smoothing(reward)
            reward = reward.squeeze(0)
        else:
            reward = self.pebal_reward

        if ood_mask.sum() > 0:
            
            reservation = torch.div(reservation, reward)
            mask = ood_mask
            # mask out each of the ood output channel
            reserve_boosting_energy = torch.add(true_pred, reservation.unsqueeze(1))[mask.unsqueeze(1).
                repeat(1, 19, 1, 1)]
            
            gambler_loss_out = torch.tensor([.0], device=logits.device)
            if reserve_boosting_energy.nelement() > 0:
                reserve_boosting_energy = torch.clamp(reserve_boosting_energy, min=1e-7).log()
                gambler_loss_out = self.ood_reg * reserve_boosting_energy

            # gambler loss for in-lier pixels
            void_mask = outlier_masks == 255
            labels[void_mask] = 0  # make void pixel to 0
            labels[mask] = 0  # make ood pixel to 0
            gambler_loss_in = torch.gather(true_pred, index=labels.unsqueeze(1), dim=1).squeeze()
            gambler_loss_in = torch.add(gambler_loss_in, reservation)

            # exclude the ood pixel mask and void pixel mask
            gambler_loss_in = gambler_loss_in[(~mask) & (~void_mask)].log()
            return {'gambler_loss': -(gambler_loss_in.mean() + gambler_loss_out.mean())}
        else:
            mask = outlier_masks == 255
            labels[mask] = 0
            reservation = torch.div(reservation, reward)
            gambler_loss = torch.gather(true_pred, index=labels.unsqueeze(1), dim=1).squeeze()
            gambler_loss = torch.add(gambler_loss, reservation)
            gambler_loss = gambler_loss[~mask].log()
            # assert not torch.any(torch.isnan(gambler_loss)), "nan check"
            return {'gambler_loss': -gambler_loss.mean()}
    
    def densehybrid_loss(self, outputs, targets, indices, num_masks):

        mask_logits = outputs["pred_masks"]
        class_logits = outputs["pred_logits"]
        class_logits = F.softmax(class_logits, dim=-1)[..., :-1]
        mask_logits = mask_logits.sigmoid()
        
        logits_ood = outputs["ood_pred"]
       
        outlier_masks = torch.cat([x["outlier_masks"].unsqueeze(
            0) for x in targets], dim=0)  # -> (B, H, W)
        
        labels = torch.cat([x["sem_seg"].unsqueeze(
            0) for x in targets], dim=0)  # -> (B, H, W)

        logits = torch.einsum("bqc,bqhw->bchw", class_logits, mask_logits)
        logits = F.interpolate(logits, size=outlier_masks.shape[-2:], mode="bilinear", align_corners=True)
        logits_ood = F.interpolate(logits_ood, size=outlier_masks.shape[-2:], mode="bilinear", align_corners=True)
        ##########################

        cls_out = F.log_softmax(logits, dim=1)
        ood_out = F.log_softmax(logits_ood, dim=1)

        label_ood = torch.zeros_like(labels)
        label_ood[labels == 254] = 1
        lse = torch.logsumexp(logits, dim=1) * label_ood
        
        use_ma = True
        if use_ma:
            reg = get_batch_avg(logits, label_ood) # useless
        else:
            reg = -logits.mean(dim=1) * label_ood
        
        loss_ood = (lse + reg.detach()).sum() / label_ood[label_ood==1].numel()
        labels[labels == 255] = self.num_classes
        labels[labels == 254] = self.num_classes
        
        loss_seg = F.nll_loss(cls_out, labels, ignore_index=self.num_classes)
        # outlier_masks[outlier_masks == 255] = 2
        loss_th = F.nll_loss(ood_out, label_ood, ignore_index=2)

        loss = loss_seg + self.densehybrid_beta * loss_ood + self.densehybrid_beta * 10 * loss_th

        return {"densehybrid_loss": loss}

    def outlier_loss(self, outputs, targets, indices, num_masks):
        """
        This loss is used with outlier supervision in order to explicitly minimize anomaly
        score for inlier pixels and maximize it for outlier pixels

        Adapted from https://github.com/tianyu0207/PEBAL/blob/main/code/losses.py
        """

        outlier_masks = torch.cat([x["outlier_masks"].unsqueeze(
            0) for x in targets], dim=0)  # -> (B, H, W)

        ood_mask = (outlier_masks == 1)
        id_mask = (outlier_masks == 0)

        mask_logits = outputs["pred_masks"]
        class_logits = outputs["pred_logits"]
        class_logits = F.softmax(class_logits, dim=-1)[..., :-1]
        mask_logits = mask_logits.sigmoid()

        logits = torch.einsum("bqc,bqhw->bchw", class_logits, mask_logits)

        if self.outlier_loss_target == "nls":
            if self.score_norm == "sigmoid":
                score = logits.sigmoid()
            elif self.score_norm == "tanh":
                score = logits.tanh()
            else:
                score = logits
            score = -score.sum(dim=1)  # -> (B, H, W)
        elif self.outlier_loss_target == "energy":
            score = -torch.logsumexp(logits, dim=1)  # -> (B, H, W)
        elif self.outlier_loss_target == "softmax_entropy":
            score = torch.special.entr(logits.softmax(dim=1)).sum(dim=1)
        elif self.outlier_loss_target == "sum_entropy":
            score = torch.special.entr(logits.div(logits.sum(dim=1, keepdims=True))).sum(dim=1)
        else:
            raise ValueError(
                f"Undefined Outlier Target Score: f{self.outlier_loss_target}")

        score = F.interpolate(score.unsqueeze(
            1), size=outlier_masks.shape[-2:], mode="bilinear", align_corners=True).squeeze(1)

        ood_score = score[ood_mask]
        id_score = score[id_mask]

        if self.outlier_loss_func == "squared_hinge":
            loss = torch.pow(
                F.relu(id_score - self.inlier_upper_threshold), 2).mean()
            if ood_mask.sum() > 0:
                loss = loss + \
                    torch.pow(
                        F.relu(self.outlier_lower_threshold - ood_score), 2).mean()
                loss = 0.5 * loss
        elif self.outlier_loss_func == "binary_cross_entropy":
            loss = 0.5 * F.binary_cross_entropy_with_logits(score, ood_mask.float()) # NOTE: try score + 1 to make it 1-\sigma(others) 

        elif self.outlier_loss_func == 'mse':
            id_up_thr_vec = torch.tensor(self.inlier_upper_threshold).to(id_score.device)
            id_up_thr_vec = id_up_thr_vec.repeat(id_score.shape)

            loss = F.mse_loss(id_score, id_up_thr_vec)

            if ood_mask.sum() > 0:
                ood_low_thr_vec = torch.tensor(self.outlier_lower_threshold).to(ood_score.device)
                ood_low_thr_vec = ood_low_thr_vec.repeat(ood_score.shape)

                loss = loss + \
                    F.mse_loss(ood_score, ood_low_thr_vec)
                loss = 0.5 * loss


        elif self.outlier_loss_func == 'l1':
            id_up_thr_vec = torch.tensor(self.inlier_upper_threshold).to(id_score.device)
            id_up_thr_vec = id_up_thr_vec.repeat(id_score.shape)

            loss = F.l1_loss(id_score, id_up_thr_vec)

            if ood_mask.sum() > 0:
                ood_low_thr_vec = torch.tensor(self.outlier_lower_threshold).to(ood_score.device)
                ood_low_thr_vec = ood_low_thr_vec.repeat(ood_score.shape)

                loss = loss + \
                    F.l1_loss(ood_score, ood_low_thr_vec)
                loss = 0.5 * loss

        elif self.outlier_loss_func == 'kl':
            score = F.interpolate(logits, size=outlier_masks.shape[-2:], mode="bilinear", align_corners=True)
            K = logits.shape[1]

            score = score.permute(0,2,3,1).reshape(-1,K)

            id_mask = id_mask.view(-1).unsqueeze(-1)
            ood_mask = ood_mask.view(-1).unsqueeze(-1)

            id_score = torch.mul(score, id_mask)
            ood_score = torch.mul(score, ood_mask)

            sorted_id = id_score.sort(dim=-1, descending=True)[0].log_softmax(dim=-1)
            sorted_id = sorted_id.log_softmax(dim=-1)
            target_id = torch.zeros_like(sorted_id)
            target_id[:,0] = 1.
            target_id = target_id.softmax(dim=-1)

            loss = F.kl_div(sorted_id, target_id)

            if ood_mask.sum() > 0:
                
                sorted_ood = ood_score.sort(dim=-1, descending=True)[0].log_softmax(dim=-1)
                target_ood = torch.zeros_like(sorted_ood).softmax(dim=-1)

                loss = loss + \
                    F.kl_div(sorted_ood, target_ood)
                loss = 0.5 * loss

        else:
            raise ValueError(
                f"Undefined Outlier Loss Function: f{self.outlier_loss_func}")

        return {"outlier_loss": loss}

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i)
                              for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i)
                              for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
            'smoothness': self.smoothness_loss,
            'sparsity': self.sparsity_loss,
            'outlier': self.outlier_loss,
            'gambler': self.gambler_loss,
            'densehybrid': self.densehybrid_loss
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k,
                               v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(
                iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(
                loss, outputs, targets, indices, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    # Dense Hybrid loss is computed only for the last layer
                    if loss == 'densehybrid':
                        continue
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
