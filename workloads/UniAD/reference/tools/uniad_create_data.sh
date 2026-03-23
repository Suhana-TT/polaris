# SPDX-FileCopyrightText: (C) 2020 SenseTime. All Rights Reserved
# SPDX-License-Identifier: Apache-2.0

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python tools/create_data.py nuscenes --root-path ./data/nuscenes \
       --out-dir ./data/infos \
       --extra-tag nuscenes \
       --version v1.0 \
       --canbus ./data/nuscenes \