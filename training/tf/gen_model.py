#!/usr/bin/env python3
import os
import sys
from tfprocess import TFProcess
import setting

blocks = 6
channels = 64
tfprocess = TFProcess(setting.RESIDUAL_BLOCKS, setting.RESIDUAL_FILTERS)
tfprocess.init(batch_size=1, gpus_num=1)
# tfprocess.replace_weights(weights)
path = os.path.join(os.getcwd(), "leelaz-model")
tfprocess.save_leelaz_weights('restored.txt')
# save_path = tfprocess.saver.save(tfprocess.session, path, global_step=0)