# SPDX-FileCopyrightText: (C) 2020 SenseTime. All Rights Reserved
# SPDX-License-Identifier: Apache-2.0

from mmcv.runner.hooks.hook import HOOKS, Hook
from ..utils import run_time


@HOOKS.register_module()
class GradChecker(Hook):

    def after_train_iter(self, runner):
        for key, val in runner.model.named_parameters():
            if val.grad is None and val.requires_grad:
                runner.logger.warning( 
                    "WARNING: {key}'s parameters are not being used!"
                ) 
