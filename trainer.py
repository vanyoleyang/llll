import os
import torch
import time
import datetime
import numpy as np
from utils import adjustLR, makeDir, displayImage, displayMask, displayHand, setCUDA, convertLossList, saveLog, saveCheckpoint, saveCheckpointBestModel


def pretrain(args, dataloaders, model, criterion, optimizer):

    print('\n Pretrain...\n')

    # Initialize Variables
    dataloader_HAND3D_train = dataloaders['HAND3D']['train']
    dataloader_HAND3D_valid = dataloaders['HAND3D']['valid']
    # dataloader_STEREO_valid = dataloaders['STEREO']['valid']
    loss_valid_best = 1000.0
    loss_valid_delta = 0.0

    # Pretrain the model
    for epoch in range(args.max_epochs_pretrain):
        # Initialize learning rate
        learning_rate = adjustLR(optimizer, epoch, args.lr_base_pretrain, policy=args.lr_policy_pretrain, policy_parameter=args.lr_policy_param_pretrain)
        # Intialize variables
        metrics = {'loss': [], 'loss_list': {'loss_2d': [], 'loss_3d': [], 'loss_mask': [], 'loss_reg': [], 'loss_camera': [], 'avg_distance_2d': [list() for _ in range(args.n_kps)], 'avg_distance_3d': [list() for _ in range(args.n_kps)]}}
        for i, (data) in enumerate(dataloader_HAND3D_train):
            # Set CUDA
            image, mask, targets, index = setCUDA(args, data)
            # Initialize optimizer
            optimizer.zero_grad()
            # Get camera_parameters
            predictions = model(image, right=True)
            # Get loss
            loss, loss_list = criterion(epoch, mask, predictions, targets)
            # Optimize the model
            loss.backward()
            optimizer.step()

            # Keep track of metrics
            metrics['loss'].append(loss.item())
            metrics['loss_list'] = convertLossList(metrics['loss_list'], loss_list)

            # Print log
            if (i+1) % 50 == 0:
                saveLog(args, epoch, args.max_epochs_pretrain, i, dataloader_HAND3D_train, learning_rate, loss, metrics, mode='Pretr')

        # Validation
        loss_HAND3D_valid = valid(args, epoch, args.max_epochs_pretrain, learning_rate, dataloader_HAND3D_valid, model, criterion, mode='Pretr', display_2D=True, display_3D=False)
        # loss_STEREO_valid = valid(args, epoch, args.max_epochs_pretrain, learning_rate, dataloader_STEREO_valid, model, criterion, mode='Pretr', display_2D=True, display_3D=False)

        # Save the model checkpoints
        if (epoch+1) % args.interval_checkpoint:
            saveCheckpoint(args, model, optimizer, pretrain=True)

        # Save the best model
        if loss_HAND3D_valid < (loss_valid_best - loss_valid_delta):
            loss_valid_best = loss_HAND3D_valid
            saveCheckpointBestModel(args, model, optimizer, pretrain=True)


def train(args, dataloaders, model, criterion, optimizer):

    print('\n Train...\n')

    # Initialize Variables
    dataloader_train = dataloaders['HAND3D']['train']
    dataloader_valid = dataloaders['HAND3D']['valid']
    loss_valid_best = 1000.

    # Train the model
    for epoch in range(args.max_epochs_train):
        # Initialize learning rate
        learning_rate = adjustLR(optimizer, epoch, args.lr_base_train, policy=args.lr_policy_train, policy_parameter=args.lr_policy_param_train)
        # learning_rate = args.lr_base_train
        # Intialize variables
        metrics = {'loss': [], 'loss_list': {'loss_2d': [], 'loss_3d': [], 'loss_mask': [], 'loss_reg': [], 'loss_camera': [], 'avg_distance_2d': [list() for _ in range(args.n_kps)], 'avg_distance_3d': [list() for _ in range(args.n_kps)]}}
        for i, (data) in enumerate(dataloader_train):
            # Set CUDA
            image, mask, targets, index = setCUDA(args, data)
            # Initialize optimizer
            optimizer.zero_grad()
            # Get camera_parameters
            predictions = model(image)
            # Get loss
            loss, loss_list = criterion(epoch, mask, predictions, targets)
            # Optimize the model
            loss.backward()
            optimizer.step()

            # Keep track of metrics
            metrics['loss'].append(loss.item())
            metrics['loss_list'] = convertLossList(metrics['loss_list'], loss_list)

            # Print log
            if (i+1) % 100 == 0:
                saveLog(args, epoch, args.max_epochs_train, i, dataloader_train, learning_rate, loss, metrics, mode='Train')
        if (epoch + 1) % 1 == 0:
            # Validation
            loss_valid = valid(args, epoch, args.max_epochs_train, learning_rate, dataloader_valid, model, criterion,
                               mode='Valid', display_2D=False, display_3D=False)
            # loss_STEREO_valid = valid(args, epoch, args.max_epochs_train, learning_rate, dataloader_STEREO_valid, model, criterion, mode='Train', display_2D=True, display_3D=False)
            # Save the best model
            if loss_valid < (loss_valid_best - 0.):
                loss_valid_best = loss_valid
                saveCheckpointBestModel(args, model, optimizer, pretrain=False)

            # Save the model checkpoints
        if (epoch+1) % 1. == 0:
                saveCheckpoint(args, model, optimizer, pretrain=False)

from statistics import mean

def valid(args, epoch, max_epochs, learning_rate, dataloader, model, criterion, mode='Pretr', display_2D=False, display_3D=False):

    # Set the model in evaluation mode
    model.eval()

    # Intialize variables
    flag_right = True
    dataset_name = type(dataloader.dataset).__name__
    if dataset_name == 'EgoDexter_seq' or dataset_name == 'DexterObject_seq':
        n_kps = 5
    else:
        n_kps = args.n_kps
    metrics = {'loss': [], 'loss_list': {'loss_2d': [], 'loss_3d': [], 'loss_mask': [], 'loss_reg': [], 'loss_camera': [], 'avg_distance_2d': [list() for _ in range(n_kps)], 'avg_distance_3d': [list() for _ in range(n_kps)]}}
    with torch.no_grad():
        for i, (data) in enumerate(dataloader):
            # Set CUDA
            image, mask, targets, index = setCUDA(args, data)
            # Get camera_parameters
            predictions = model(image)
            # Get loss
            loss, loss_list = criterion(epoch, mask, predictions, targets, train=False)

            # Keep track of metrics
            metrics['loss'].append(loss.item())
            metrics['loss_list'] = convertLossList(metrics['loss_list'], loss_list)
            # if display_2D:
            #     displayImage(args, epoch, i, image, predictions, targets, '')
            # Print log
            if (i + 1) == len(dataloader):
                saveLog(args, epoch, max_epochs, i, dataloader, learning_rate, loss, metrics, mode='Valid')
            #     str = mode
            #     if display_2D:
            #         displayImage(args, epoch, i, image, predictions, targets, '')
            #         displayMask(args, epoch, i, mask, predictions, '')
                # if display_3D and n_kps == 21:
                #     displayHand(args, epoch, i, predictions, targets, '')
    # Set the model in training mode
    model.train()
    ll = loss.item() - mean(metrics['loss_list']['loss_reg'])
    # print(loss.item() - metrics['loss_list']['loss_reg'])
    return ll
