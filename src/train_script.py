#!/usr/bin/python
# -*- coding: utf-8 -*-
"""scripts for training models sequentially
"""

import os
import itertools
import subprocess


base_cmd = ["python3", "src/main.py",
            "--gpu", "9"]

config_dir = "config"
configs = ["mlp.yml", "mlp_mda.yml", "mlp_mda_multitask.yml",
           "mlp_mda_fe.yml", "gbdt.yml", "gbdt_mda_cv_multitask.yml"]
modes = ["train"]

parallel = 1

cmds = []
for m, c in itertools.product(modes, configs):
    if m in ["train", "test"] and os.path.splitext(c)[1] != ".yml":
        continue
    cmd = base_cmd + ["--config_file", os.path.join(config_dir, c), 
                      "--mode", m]
    cmds.append(cmd)

for idx in range(0, len(cmds), parallel):
    cmds_para = cmds[idx:idx + parallel]
    [print(" ".join(cmd)) for cmd in cmds_para]
    procs = [subprocess.Popen(cmd) for cmd in cmds_para]
    [p.wait() for p in procs]
