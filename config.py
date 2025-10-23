# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------'

import os
import yaml
from yacs.config import CfgNode as CN
import sys

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()

_C.DATA.BATCH_SIZE = 2
_C.DATA.DATA_PATH = './dataset'
_C.DATA.IMG_SIZE = 1024
_C.DATA.PIN_MEMORY = True
_C.DATA.NUM_WORKERS = 8
# -----------------------------------------------------------------------------
# Hardware settings
# -----------------------------------------------------------------------------
_C.HARDWARE = CN()

_C.HARDWARE.N_GPU = 1 # number of gpu

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()

_C.MODEL.TYPE = 'swin'
_C.MODEL.NAME = 'swin_b'
_C.MODEL.PRETRAIN_WEIGHTS = 'segface' # segface or imagenet1k
_C.MODEL.PRETRAIN_CKPT = './pretrained_ckpt/swin_b.pth' # Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.PRETRAIN_SEGFACE = './network/pretrained_weights/SegFace_swin_celaba_512.pt' # path to segface weights
_C.MODEL.PRETRAIN_IMAGENET1K = './network/pretrained_weights/swin_b-68c6b09e.pth' # path to segface weights

_C.MODEL.NUM_CLASSES = 1 # Number of classes, overwritten in data preparation
_C.MODEL.DROP_RATE = 0.0 # Dropout rate
_C.MODEL.DROP_PATH_RATE = 0.1 # Drop path rate
_C.MODEL.LABEL_SMOOTHING = 0.1 # Label Smoothing

_C.MODEL.FREEZE_ENCODER = True #Encoder Freenzing
_C.MODEL.STAGE3_UNFREEZE_PERIODE = 0.4 # in percent (0: no freezing, 1: all epochs are freezed)
_C.MODEL.STAGE2_UNFREEZE_PERIODE = 0.7 # How long should the Encoder be freezed
_C.MODEL.STAGE1_UNFREEZE_PERIODE = 0.9
_C.MODEL.STAGE0_UNFREEZE_PERIODE = 0.98

# -----------------------------------------------------------------------------
# Swin Transformer settings
# -----------------------------------------------------------------------------
_C.MODEL.SWIN = CN()

_C.MODEL.SWIN.PATCH_SIZE = 4
_C.MODEL.SWIN.IN_CHANS = 3
_C.MODEL.SWIN.EMBED_DIM = 128
_C.MODEL.SWIN.DEPTHS = [2, 2, 18, 2]
_C.MODEL.SWIN.DECODER_DEPTHS = [2, 2, 6, 2] # for symetrie [2, 2, 18, 2]
_C.MODEL.SWIN.NUM_HEADS = [4, 8, 16, 32]
_C.MODEL.SWIN.WINDOW_SIZE = 7
_C.MODEL.SWIN.MLP_RATIO = 4.
_C.MODEL.SWIN.QKV_BIAS = True
_C.MODEL.SWIN.QK_SCALE = None
_C.MODEL.SWIN.APE = False
_C.MODEL.SWIN.PATCH_NORM = True
_C.MODEL.SWIN.FINAL_UPSAMPLE= "expand_first"

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------

_C.TRAIN = CN()
_C.TRAIN.MAX_EPOCHS = 300
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 0
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6

_C.TRAIN.ACCUMULATION_STEPS = 1 # Gradient accumulation steps # could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False # Whether to use gradient checkpointing to save memory

# Tversky Loss
_C.TRAIN.TVERSKY_LOSS_ALPHA = 0.4
_C.TRAIN.TVERSKY_LOSS_BETA = 0.6
# Unified Focal Loss
_C.TRAIN.UF_LOSS_DELTA = 0.6
_C.TRAIN.UF_LOSS_GAMMA = 0.5
_C.TRAIN.UF_LOSS_WEIGTH = 0.5

_C.TRAIN.EARLY_STOPPING_PATIENCE = 15
_C.TRAIN.EARLY_STOPPING_FLAG = False
_C.TRAIN.SIG_THRESHOLD = 0.5

# -----------------------------------------------------------------------------
# LR_SCHEDULER
# -----------------------------------------------------------------------------
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
_C.TRAIN.LR_SCHEDULER.WARMUP_PREFIX = True 

# -----------------------------------------------------------------------------
# Optimizer
# -----------------------------------------------------------------------------
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
_C.TRAIN.OPTIMIZER.EPS = 1e-8 # Optimizer Epsilon
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999) # Optimizer Betas

# -----------------------------------------------------------------------------
# Test settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
_C.TEST.SIG_THRESHOLD = 0.5 # threshold for the gereration of the binary mask for the validation

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
_C.OUTPUT_DIR = './model_out'
_C.LIST_DIR ='./lists'
_C.SEED = 1234 # Fixed random seed
_C.DETERMINISTIC = True
_C.SAVE_BEST_RUN = False
_C.SAVE_LAST_RUN = False
#_C.LOCAL_RANK = 0 #only if more than one gpu
# -----------------------------------------------------------------------------

def _update_config_from_file(config, cfg_file):
    config.defrost()
    if cfg_file != 'None':
        with open(cfg_file, 'r') as f:
            yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

        for cfg in yaml_cfg.setdefault('BASE', ['']):
            if cfg:
                _update_config_from_file(
                    config, os.path.join(os.path.dirname(cfg_file), cfg)
                )
        print('=> merge config from {}'.format(cfg_file), file=sys.stderr)
        config.merge_from_file(cfg_file)
    else:
        raise ValueError("config file not found")
    config.freeze()


def update_config(config, bool_test, bool_train, args):
    _update_config_from_file(config, args.cfg)
    if bool_test and bool_train:
        raise ValueError(f"test and train flags are rised incorectly (Both true)!")
    if not bool_test and not bool_train:
        raise ValueError(f"test and train flags are rised incorectly (Both false)!")
    # merge from specific arguments
    config.defrost()

    if args.output_dir:
        config.OUTPUT_DIR = args.output_dir
    if args.use_checkpoint:
        config.TRAIN.USE_CHECKPOINT = True
    if args.deterministic:
        config.DETERMINISTIC = True
    if args.weight_decay:
        config.TRAIN.WEIGHT_DECAY = args.weight_decay

    if bool_train:
        if args.sig_threshold:
            config.TRAIN.SIG_THRESHOLD = args.sig_threshold

    if bool_test:
        if args.sig_threshold:
            config.TEST.SIG_THRESHOLD = args.sig_threshold
    
    config.freeze()


def get_config(args,bool_train, bool_test):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    if args != None:
        update_config(config, bool_test, bool_train, args)
    else:
        raise ValueError("no argumens given")
    return config
