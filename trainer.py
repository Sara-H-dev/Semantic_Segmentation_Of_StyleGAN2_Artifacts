# from trainer_Synapse.py

import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
#from utils import DiceLoss, ELoss
from torchvision import transforms

def trainer_MS_UNet(args, model, log_save_path = ""):
    from dataset.dataset import SegArtifact_dataset, RandomGenerator

    # logger config
    logging.basicConfig(
        filename = log_save_path + "/log.txt",
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s', 
        datefmt='%H:%M:%S') #houres, minutes, seconds
    # every log is visible in terminal and log-file
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args)) # logs the args

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

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(   db_train, 
                                batch_size = batch_size, 
                                shuffle = True,                     # shuffles the data sequence
                                num_workers = 8,                    # number of parallel processes (Number_of_CPU_cores / 2)
                                pin_memory = True,                  # true if GPU available
                                worker_init_fn = worker_init_fn )   # worker seed
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    
    # training!!!
    model.train()

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    edge_loss = ELoss(num_classes)

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
    # creates progress bar
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        # get the batches (image & label) from the DataLoader
        for i_batch, sampled_batch in enumerate(trainloader):

            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda() # moves to GPUs

            outputs = model(image_batch)
            
            # loss
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss_edge = edge_loss(outputs, edge_batch, softmax=True)
            # mixing losses
            if epoch_num >= 50:
                loss = loss_dice * 0.5+ loss_ce * 0.5 +loss_edge * 0.1
            else:
                loss = loss_dice * 0.5+ loss_ce * 0.5
            
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
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))
        
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
