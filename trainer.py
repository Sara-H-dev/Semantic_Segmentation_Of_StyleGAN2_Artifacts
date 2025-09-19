# from trainer_Synapse.py

import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from loss.TverskyLoss import TverskyLoss_binary

def worker_init_fn(worker_id, seed):
        random.seed(seed + worker_id)

def trainer_MS_UNet(args, model, log_save_path = "", config = None):
    from dataset.dataset import SegArtifact_dataset, RandomGenerator

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # logger config
    logging.basicConfig(
        filename = log_save_path + "/log.txt",
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s', 
        datefmt='%H:%M:%S') #houres, minutes, seconds
    # every log is visible in terminal and log-file
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args)) # logs the args

    if config == None: raise ValueError(logging.error("Config file is not found!"))
    freeze_encoder = config.MODEL.FREEZE_ENCODER

    base_lr = args.base_lr                      # learning rate
    num_classes = args.num_classes              # number of classes
    batch_size = args.batch_size * args.n_gpu   # batch_size
    # max_iterations = args.max_iterations

    # preparation of data
    db_train = SegArtifact_dataset( base_dir = args.root_path, 
                                    list_dir = args.list_dir, 
                                    split = "train",
                                    transform = transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    
    print("The length of train set is: {}".format(len(db_train)))

    trainloader = DataLoader(   db_train, 
                                batch_size = batch_size, 
                                shuffle = True,                     # shuffles the data sequence
                                num_workers = 8,                    # number of parallel processes (Number_of_CPU_cores / 2)
                                pin_memory = torch.cuda.is_available(),                  # true if GPU available
                                worker_init_fn = worker_init_fn(0,seed=args.seed) )   # worker seed
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    
    # freez encoder if wanted
    model.freeze_encoder(freeze_encoder)
    # training!!!
    model.train()

    tversky_loss = TverskyLoss_binary(config.TRAIN.TVERSKY_LOSS_ALPHA, config.TRAIN.TVERSKY_LOSS_BETA)

    # Stochastic Gradient Decent
    optimizer = optim.SGD(  model.parameters(), 
                            lr = base_lr,           # learning rate
                            momentum = 0.9,         # accelerates updates towards the gradient direction (prevents zick, zack)
                            weight_decay=0.0001)    # L2 regularisation. 
                                                    # keeps weights small → reduces overfitting.
                                                    
    writer = SummaryWriter(log_save_path + '/log')

    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    # parameters for unfreezing the encoder:
    stage3_unfreeze = max_epoch*config.MODEL.STAGE3_UNFREEZE_PERIODE
    stage2_unfreeze = max_epoch*config.MODEL.STAGE2_UNFREEZE_PERIODE
    stage1_unfreeze = max_epoch*config.MODEL.STAGE1_UNFREEZE_PERIODE
    stage0_unfreeze = max_epoch*config.MODEL.STAGE1_UNFREEZE_PERIODE
    # creates progress bar
    iterator = tqdm(range(max_epoch), ncols=70,  dynamic_ncols=True)

    bool_s3_unfreezed = False
    bool_s2_unfreezed = False
    bool_s1_unfreezed = False
    bool_s0_unfreezed = False

    for epoch_num in iterator:
        if freeze_encoder == True:
            # unfreeze form the deepest encoder level to the highests
            if (epoch_num >= stage3_unfreeze) and (bool_s3_unfreezed == False):
                model.unfreeze_encoder(3)
                bool_s3_unfreezed = True
            if (epoch_num >= stage2_unfreeze) and (bool_s2_unfreezed == False):
                model.unfreeze_encoder(2)
                bool_s2_unfreezed = True
            if (epoch_num >= stage1_unfreeze) and (bool_s1_unfreezed == False):
                model.unfreeze_encoder(1)
                bool_s1_unfreezed = True
            if (epoch_num >= stage0_unfreeze) and (bool_s0_unfreezed == False):
                model.unfreeze_encoder(0)
                bool_s0_unfreezed = True

        # get the batches (image & label) from the DataLoader
        for i_batch, sampled_batch in enumerate(trainloader):

            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.to(device), label_batch.to(device) # moves to GPUs

            print("before model \n")

            outputs = model(image_batch)

            print("after model \n")
            if i_batch == 0:  # nur einmal beim ersten Batch
                with torch.no_grad():
                    msg = (f"[CHECK E{epoch_num:03d}] "
                       f"outputs.shape={tuple(outputs.shape)}, dtype={outputs.dtype}, "
                       f"min={outputs.min().item():.4f}, max={outputs.max().item():.4f}, "
                       f"labels={torch.unique(label_batch).tolist()}  (dtype={label_batch.dtype})")
                    tqdm.write(msg)
            
            # loss
            loss_tversky = tversky_loss.forward(outputs, label_batch)
            loss = loss_tversky

            # backprop + optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # poly-learning rate
            # learning rate slowly drops to 0 
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            # iter and logging
            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))
        
        # saves all 50 epochs
        save_interval = 50 

        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(log_save_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save({    'epoch': epoch_num,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'iter_num': iter_num,}, save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        # saves the last run
        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(log_save_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save({    'epoch': epoch_num,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'iter_num': iter_num,}, save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"
