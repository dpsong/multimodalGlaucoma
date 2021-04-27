"""
Multimodal glaucoma classification training Script.
This script runs the training.
You may need to write your own script with your datasets and other customizations.
"""

import os
import torch
import random
import shutil
import argparse
import numpy as np
from torch.utils.data import DataLoader

import _init_paths
from datalayer.octvfdatalayer import OctVfDataset
from vfoct_attention import VfOctAttentionNet


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-seed', type=int, default=6)
    parser.add_argument('-bs', type=int, default=32)
    parser.add_argument('-lr', type=float, default=0.001)
    parser.add_argument('-deviceID', type=int, default=0)
    parser.add_argument('-epochs', type=int, default=120)
    parser.add_argument('-filename', type=str, default='checkpoints/VFOCT-')
    return parser.parse_args()


def get_data_loader(root_dir, textlist_name, batch_size=32, shuffle=False):
    dataset = OctVfDataset(root_dir, textlist_name)
    if batch_size is None:
        batch_size = int(dataset.__len__() / 1)

    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8)

    return data_loader


def save_checkpoint(state, args):
    filename = args.filename + '_{}.pth'.format(state['epoch'])
    torch.save(state, filename)


def train(net, trainloader, criterion, optimizer, epoch, device='cpu'):

    net.train()
    running_loss = 0.0
    correct = 0
    total = 0
    print('Start EP-{} training'.format(epoch + 1))

    for i_batch, sample_batched in enumerate(trainloader, 0):
        oct_datas = sample_batched['oct_data']
        vf_datas = sample_batched['vf_data']
        labels = sample_batched['label']
        
        oct_datas = oct_datas.to(device)
        vf_datas = vf_datas.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = net(oct_datas, vf_datas)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i_batch % 5 == 4:  
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i_batch + 1, running_loss / 5))
            running_loss = 0.0

        # accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 1.0 * correct / total
    print('EP-{} accuracy on train_dataset: {}'.format(epoch + 1, accuracy))


def main():
    args = get_args()
    # set random seeds
    fix_seed(args.seed)
    device = torch.device("cuda:{}".format(args.deviceID) if torch.cuda.
                          is_available() else "cpu")

    net = VfOctAttentionNet()
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(
        net.parameters(), lr=args.lr, eps=1e-08, weight_decay=0.0005)

    # Dataloader
    root_dir = 'data/vfoct_data'
    train_text = 'data/train.txt'
    train_loader = get_data_loader(
        root_dir, train_text, batch_size=args.bs, shuffle=True)
        
    # Training
    for epoch in range(args.epochs):
        train(net, train_loader, criterion, optimizer, epoch, device)
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'filename': args.filename,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, args)

    print('Finished Training')


if __name__ == '__main__':
    main()
