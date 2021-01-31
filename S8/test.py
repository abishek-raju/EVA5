#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 11:49:34 2021

@author: rampfire
"""
import torch
from tqdm import tqdm

def test(epoch,net,testloader,device,criterion):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        with tqdm(total=len(testloader)) as pbar:
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)
    
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                # print(f'batch_idx:{batch_idx},loss:{(test_loss/(batch_idx+1))},acc:{100.*correct/total}')
                pbar.update(1)
                # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    print(f'\nEpoch: {epoch} Train set: Average loss: {(test_loss/(batch_idx+1))}, Accuracy: {100.*correct/total}%')
