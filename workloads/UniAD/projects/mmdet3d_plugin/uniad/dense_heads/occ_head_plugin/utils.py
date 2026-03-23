# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import numpy as np


def calculate_birds_eye_view_parameters(x_bounds, y_bounds, z_bounds):
    bev_resolution = np.array([row[2] for row in [x_bounds, y_bounds, z_bounds]])
    bev_start_position = np.array(
        [row[0] + row[2] / 2.0 for row in [x_bounds, y_bounds, z_bounds]]
    )
    bev_dimension = np.array(
        [(row[1] - row[0]) / row[2] for row in [x_bounds, y_bounds, z_bounds]],
        dtype=np.int64,
    )
    return bev_resolution, bev_start_position, bev_dimension


def gen_dx_bx(xbound, ybound, zbound):
    dx = np.array([row[2] for row in [xbound, ybound, zbound]])
    bx = np.array([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = np.array(
        [(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]], dtype=np.int64
    )
    return dx, bx, nx


def update_instance_ids(instance_seg, old_ids, new_ids):
    indices = np.arange(old_ids.max() + 1)
    for old_id, new_id in zip(old_ids, new_ids):
        indices[old_id] = new_id
    return indices[instance_seg].astype(np.int64)


def make_instance_seg_consecutive(instance_seg):
    unique_ids = np.unique(instance_seg)
    new_ids = np.arange(len(unique_ids))
    instance_seg = update_instance_ids(instance_seg, unique_ids, new_ids)
    return instance_seg


def predict_instance_segmentation_and_trajectories(
    foreground_masks, ins_sigmoid, vehicles_id=1
):
    if foreground_masks.ndim == 5 and foreground_masks.shape[2] == 1:
        foreground_masks = foreground_masks.squeeze(2)
    foreground_masks = foreground_masks == vehicles_id
    argmax_ins = np.argmax(ins_sigmoid, axis=1)
    argmax_ins = argmax_ins + 1
    instance_seg = (argmax_ins * foreground_masks.astype(float)).astype(np.int64)
    instance_seg = make_instance_seg_consecutive(instance_seg).astype(np.int64)
    return instance_seg
