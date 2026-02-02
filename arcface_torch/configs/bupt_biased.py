from easydict import EasyDict as edict
import os
import datetime

config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "r50"
config.resume = False
config.output = None
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 512
config.lr = 0.1
config.verbose = 2000
config.dali = False

config.rec = os.environ["DATA_DIR"] + "/bupt-balancedface-mxnet_biased_African_10pct/flattened__Caucasian_African"
config.num_classes = 7700
config.num_image = 359021
config.num_epoch = 20
config.warmup_epoch = 0
config.val_targets = ['African_test', 'Asian_test', 'Caucasian_test', 'Indian_test']
config.using_wandb = False

dt = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
config.output = os.environ["ARTIFACT_DIR"] + f"/trained/arcface_torch/r50/bupt-balancedface-mxnet_biased_African_10pct/Caucasian_African/{dt}/"