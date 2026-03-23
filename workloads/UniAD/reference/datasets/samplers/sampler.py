# SPDX-FileCopyrightText: (C) 2020 SenseTime. All Rights Reserved
# SPDX-License-Identifier: Apache-2.0

from mmcv.utils.registry import Registry, build_from_cfg

SAMPLER = Registry("sampler")


def build_sampler(cfg, default_args):
    return build_from_cfg(cfg, SAMPLER, default_args)
