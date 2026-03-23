# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTSim: occ_head_plugin/metrics.py — Pure Python/numpy replacement.
pytorch_lightning.Metric base class replaced with a plain Python class.
"""

from typing import Optional
import numpy as np


class _Metric:
    """Plain Python base class replacing pytorch_lightning.metrics.Metric."""

    def __init__(self, compute_on_step: bool = False):
        self._states: dict = {}

    def add_state(self, name: str, default, dist_reduce_fx=None):
        self._states[name] = (
            np.array(default) if not isinstance(default, np.ndarray) else default.copy()
        )
        setattr(self, name, self._states[name])

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError

    def reset(self):
        for name, default in self._states.items():
            setattr(self, name, default.copy())


class IntersectionOverUnion(_Metric):
    """Computes intersection-over-union."""

    true_positive: np.ndarray
    false_positive: np.ndarray
    false_negative: np.ndarray
    support: np.ndarray

    def __init__(
        self,
        n_classes: int,
        ignore_index: Optional[int] = None,
        absent_score: float = 0.0,
        reduction: str = "none",
        compute_on_step: bool = False,
    ):
        super().__init__(compute_on_step=compute_on_step)
        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.absent_score = absent_score
        self.reduction = reduction

        self.add_state("true_positive", default=np.zeros(n_classes))
        self.add_state("false_positive", default=np.zeros(n_classes))
        self.add_state("false_negative", default=np.zeros(n_classes))
        self.add_state("support", default=np.zeros(n_classes))

    def update(self, prediction: np.ndarray, target: np.ndarray):
        prediction = np.asarray(prediction)
        target = np.asarray(target)
        for c in range(self.n_classes):
            pred_c = prediction == c
            tgt_c = target == c
            self.true_positive[c] += np.sum(pred_c & tgt_c)
            self.false_positive[c] += np.sum(pred_c & ~tgt_c)
            self.false_negative[c] += np.sum(~pred_c & tgt_c)
            self.support[c] += np.sum(tgt_c)

    def compute(self):
        scores = np.zeros(self.n_classes, dtype=np.float32)
        for class_idx in range(self.n_classes):
            if class_idx == self.ignore_index:
                continue
            tp = self.true_positive[class_idx]
            fp = self.false_positive[class_idx]
            fn = self.false_negative[class_idx]
            sup = self.support[class_idx]
            if sup + tp + fp == 0:
                scores[class_idx] = self.absent_score
                continue
            denominator = tp + fp + fn
            scores[class_idx] = float(tp) / denominator if denominator > 0 else 0.0

        if (self.ignore_index is not None) and (
            0 <= self.ignore_index < self.n_classes
        ):
            scores = np.concatenate([scores[: self.ignore_index], scores[self.ignore_index + 1 :]])  # type: ignore[assignment]

        if self.reduction == "mean":
            return float(np.mean(scores))
        elif self.reduction == "sum":
            return float(np.sum(scores))
        return scores


class PanopticMetric(_Metric):
    iou: np.ndarray
    true_positive: np.ndarray
    false_positive: np.ndarray
    false_negative: np.ndarray

    def __init__(
        self,
        n_classes: int,
        temporally_consistent: bool = True,
        vehicles_id: int = 1,
        compute_on_step: bool = False,
    ):
        super().__init__(compute_on_step=compute_on_step)
        self.n_classes = n_classes
        self.temporally_consistent = temporally_consistent
        self.vehicles_id = vehicles_id
        self.keys = ["iou", "true_positive", "false_positive", "false_negative"]

        self.add_state("iou", default=np.zeros(n_classes))
        self.add_state("true_positive", default=np.zeros(n_classes))
        self.add_state("false_positive", default=np.zeros(n_classes))
        self.add_state("false_negative", default=np.zeros(n_classes))

    def update(self, pred_instance, gt_instance):
        pred_instance = np.asarray(pred_instance)
        gt_instance = np.asarray(gt_instance)
        batch_size, sequence_length = gt_instance.shape[:2]
        assert gt_instance.min() == 0, "ID 0 of gt_instance must be background"
        pred_segmentation = (pred_instance > 0).astype(np.int64)
        gt_segmentation = (gt_instance > 0).astype(np.int64)

        for b in range(batch_size):
            unique_id_mapping: dict[int, int] = {}
            for t in range(sequence_length):
                result = self.panoptic_metrics(
                    pred_segmentation[b, t],
                    pred_instance[b, t],
                    gt_segmentation[b, t],
                    gt_instance[b, t],
                    unique_id_mapping,
                )
                self.iou += result["iou"]
                self.true_positive += result["true_positive"]
                self.false_positive += result["false_positive"]
                self.false_negative += result["false_negative"]

    def compute(self):
        denominator = np.maximum(
            self.true_positive + self.false_positive / 2 + self.false_negative / 2,
            np.ones_like(self.true_positive),
        )
        pq = self.iou / denominator
        sq = self.iou / np.maximum(self.true_positive, np.ones_like(self.true_positive))
        rq = self.true_positive / denominator
        return {
            "pq": pq,
            "sq": sq,
            "rq": rq,
            "denominator": (
                self.true_positive + self.false_positive / 2 + self.false_negative / 2
            ),
        }

    def panoptic_metrics(
        self,
        pred_segmentation,
        pred_instance,
        gt_segmentation,
        gt_instance,
        unique_id_mapping,
    ):
        n_classes = self.n_classes
        result = {key: np.zeros(n_classes, dtype=np.float32) for key in self.keys}

        pred_segmentation = np.asarray(pred_segmentation)
        pred_instance = np.asarray(pred_instance)
        gt_segmentation = np.asarray(gt_segmentation)
        gt_instance = np.asarray(gt_instance)

        assert pred_segmentation.ndim == 2
        assert (
            pred_segmentation.shape
            == pred_instance.shape
            == gt_segmentation.shape
            == gt_instance.shape
        )

        n_instances = int(
            np.concatenate([pred_instance.ravel(), gt_instance.ravel()]).max()
        )
        n_all_things = n_instances + n_classes
        n_things_and_void = n_all_things + 1

        prediction, pred_to_cls = self.combine_mask(
            pred_segmentation, pred_instance, n_classes, n_all_things
        )
        target, target_to_cls = self.combine_mask(
            gt_segmentation, gt_instance, n_classes, n_all_things
        )

        x = prediction + n_things_and_void * target
        bincount_2d = np.bincount(
            x.astype(np.int64).ravel(), minlength=n_things_and_void**2
        )
        conf = bincount_2d.reshape((n_things_and_void, n_things_and_void))
        conf = conf[1:, 1:]  # type: ignore[assignment]

        union = conf.sum(0, keepdims=True) + conf.sum(1, keepdims=True) - conf
        iou = np.where(
            union > 0,
            (conf.astype(float) + 1e-9) / (union.astype(float) + 1e-9),
            np.zeros_like(union, dtype=float),
        )

        mapping = np.argwhere(iou > 0.5)
        is_matching = pred_to_cls[mapping[:, 1]] == target_to_cls[mapping[:, 0]]
        mapping = mapping[is_matching]

        tp_mask = np.zeros_like(conf, dtype=bool)
        for row in mapping:
            tp_mask[row[0], row[1]] = True

        for target_id, pred_id in mapping:
            cls_id = pred_to_cls[pred_id]
            if self.temporally_consistent and cls_id == self.vehicles_id:
                if (
                    target_id in unique_id_mapping
                    and unique_id_mapping[target_id] != pred_id
                ):
                    result["false_negative"][target_to_cls[target_id]] += 1
                    result["false_positive"][pred_to_cls[pred_id]] += 1
                    unique_id_mapping[target_id] = pred_id
                    continue
            result["true_positive"][cls_id] += 1
            result["iou"][cls_id] += iou[target_id][pred_id]
            unique_id_mapping[target_id] = pred_id

        for target_id in range(n_classes, n_all_things):
            if tp_mask[target_id, n_classes:].any():
                continue
            if target_to_cls[target_id] != -1:
                result["false_negative"][target_to_cls[target_id]] += 1

        for pred_id in range(n_classes, n_all_things):
            if tp_mask[n_classes:, pred_id].any():
                continue
            if pred_to_cls[pred_id] != -1 and (conf[:, pred_id] > 0).any():
                result["false_positive"][pred_to_cls[pred_id]] += 1

        return result

    def combine_mask(self, segmentation, instance, n_classes, n_all_things):
        instance = instance.ravel().copy()
        instance_mask = instance > 0
        instance = instance - 1 + n_classes

        segmentation = segmentation.ravel().copy()
        segmentation_mask = segmentation < n_classes

        tuples_idx = np.where(instance_mask & segmentation_mask)[0]
        instance_id_to_class = -np.ones(n_all_things, dtype=np.int64)
        if len(tuples_idx) > 0:
            inst_ids = instance[tuples_idx]
            seg_cls = segmentation[tuples_idx]
            for iid, sc in zip(inst_ids, seg_cls):
                if 0 <= iid < n_all_things:
                    instance_id_to_class[iid] = sc
        for c in range(n_classes):
            instance_id_to_class[c] = c

        segmentation[instance_mask] = instance[instance_mask]
        segmentation += 1
        segmentation[~segmentation_mask] = 0

        return segmentation, instance_id_to_class
