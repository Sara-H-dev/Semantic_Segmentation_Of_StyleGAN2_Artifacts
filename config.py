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
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 4
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = './dataset'
# Dataset name
_C.DATA.DATASET = 'SegArtifact'
# Input image size
_C.DATA.IMG_SIZE = 1024
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = 'bicubic'
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 8

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = 'swin'
# Model name
_C.MODEL.NAME = 'swin_b'
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.PRETRAIN_CKPT = './pretrained_ckpt/swin_b.pth'
# path to segface weights
_C.MODEL.PRETRAIN_SEGFACE = './network/pretrained_weights/SegFace_swin_celaba_512.pt'

# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 1
# Dropout rate
_C.MODEL.DROP_RATE = 0.0
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.1
# Label Smoothing
_C.MODEL.LABEL_SMOOTHING = 0.1

#Encoder Freenzing
_C.MODEL.FREEZE_ENCODER = True
# How long should the Encoder be freezed
# in percent (0: no freezing, 1: all epochs are freezed)
_C.MODEL.STAGE3_UNFREEZE_PERIODE = 0.4
_C.MODEL.STAGE2_UNFREEZE_PERIODE = 0.7
_C.MODEL.STAGE1_UNFREEZE_PERIODE = 0.9
_C.MODEL.STAGE0_UNFREEZE_PERIODE = 0.98

# Swin Transformer parameters
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
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS = 20

_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0

# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 0
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
_C.TRAIN.LR_SCHEDULER.WARMUP_PREFIX = True 

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# Tversky Loss
_C.TRAIN.TVERSKY_LOSS_ALPHA = 0.4
_C.TRAIN.TVERSKY_LOSS_BETA = 0.6


# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Mixed precision opt level, if O0, no amp is used ('O0', 'O1', 'O2')
# overwritten by command line argument
_C.AMP_OPT_LEVEL = ''
# Path to output folder, overwritten by command line argument
_C.OUTPUT = ''
# Fixed random seed
_C.SEED = 0
# local rank for DistributedDataParallel, given by command line argument
_C.LOCAL_RANK = 0


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
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    # merge from specific arguments
    if args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
    if args.seed:
        config.SEED = args.seed
    if args.output_dir:
        config.OUTPUT = args.output_dir
    if args.accumulation_steps:
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if args.use_checkpoint:
        config.TRAIN.USE_CHECKPOINT = True
    if args.amp_opt_level:
        config.AMP_OPT_LEVEL = args.amp_opt_level
    if args.loss_alpha:
        config.TRAIN.TVERSKY_LOSS_ALPHA = args.loss_alpha
    if args.loss_beta:
         config.TRAIN.TVERSKY_LOSS_BETA = args.loss_beta
    if args.unfreeze_stage3:
        config.MODEL.STAGE3_UNFREEZE_PERIODE = args.unfreeze_stage3
    if args.unfreeze_stage2:
        config.MODEL.STAGE2_UNFREEZE_PERIODE = args.unfreeze_stage2
    if args.unfreeze_stage1:
        config.MODEL.STAGE1_UNFREEZE_PERIODE = args.unfreeze_stage1
    if args.unfreeze_stage0:
        config.MODEL.STAGE0_UNFREEZE_PERIODE = args.unfreeze_stage0 
    if args.weight_decay:
        config.TRAIN.WEIGHT_DECAY = args.weight_decay
    if args.momentum:
        config.TRAIN.OPTIMIZER.MOMENTUM = args.momentum

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    if args != None:
        update_config(config, args)

    return config
