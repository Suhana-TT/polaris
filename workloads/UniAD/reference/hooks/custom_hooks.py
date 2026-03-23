# SPDX-FileCopyrightText: (C) 2020 SenseTime. All Rights Reserved
# SPDX-License-Identifier: Apache-2.0

from mmcv.runner.hooks.hook import HOOKS, Hook


@HOOKS.register_module()
class TransferWeight(Hook):

    def __init__(self, every_n_inters=1):
        # Interval (in iterations) at which to transfer weights
        self.every_n_inters = every_n_inters

    def after_train_iter(self, runner):
        if self.every_n_iters(runner, self.every_n_inters):
            runner.eval_model.load_state_dict(runner.model.state_dict())
