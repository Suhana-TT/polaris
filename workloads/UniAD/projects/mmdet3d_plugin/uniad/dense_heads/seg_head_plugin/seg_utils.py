# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
def IOU(inputs, targets):
    numerator = (inputs * targets).sum(dim=1)
    denominator = inputs.sum(dim=1) + targets.sum(dim=1) - numerator
    loss = numerator / (denominator + 0.0000000000001)
    return loss, numerator, denominator
