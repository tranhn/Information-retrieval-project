import argparse

import pandas as pd
import torch
from thop import profile, clever_format
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import Model, set_bn_eval
# from utils import recall, LabelSmoothingCrossEntropyLoss, BatchHardTripletLoss, ImageReader, MPerClassSampler
from my_utils import recall, LabelSmoothingCrossEntropyLoss, BatchHardTripletLoss, ImageReader, MPerClassSampler

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

data_path = './dataset'
data_name = 'vn_food'
batch_size = 16
backbone_type = 'wide_resnet50_2'
gd_config = 'SG'
feature_dim = 512
num_epochs = 15
smoothing = 0.1
temperature = 0.5
margin = 0.1
recalls = 1,2,4,8

def train(net, optim, train_data_loader):
    print('starting train...')
    net.train()
    # fix bn on backbone network
    net.apply(set_bn_eval)
    total_loss, total_correct, total_num, data_bar = 0, 0, 0, train_data_loader
    for inputs, labels in data_bar:
        # inputs, labels = inputs.cuda(), labels.cuda()
        inputs, labels = inputs.to(device), labels.to(device)
        features, classes = net(inputs)
        class_loss = class_criterion(classes, labels)
        feature_loss = feature_criterion(features, labels)
        loss = class_loss + feature_loss
        optim.zero_grad()
        loss.backward()
        optim.step()
        pred = torch.argmax(classes, dim=-1)
        total_loss += loss.item() * inputs.size(0)
        total_correct += torch.sum(pred == labels).item()
        total_num += inputs.size(0)
        print('Train Epoch {}/{} - Loss:{:.4f} - Acc:{:.2f}%'
                                 .format(epoch, num_epochs, total_loss / total_num, total_correct / total_num * 100))

    return total_loss / total_num, total_correct / total_num * 100

def test(net, recall_ids):
    net.eval()
    with torch.no_grad():
        # obtain feature vectors for all data
        for key in eval_dict.keys():
            eval_dict[key]['features'] = []
            for inputs, labels in tqdm(eval_dict[key]['data_loader'], desc='processing {} data'.format(key)):
                # inputs, labels = inputs.cuda(), labels.cuda()
                inputs, labels = inputs.to(device), labels.to(device)
                features, classes = net(inputs)
                eval_dict[key]['features'].append(features)
            eval_dict[key]['features'] = torch.cat(eval_dict[key]['features'], dim=0)

        # compute recall metric
        if data_name == 'isc':
            acc_list = recall(eval_dict['test']['features'], test_data_set.labels, recall_ids,
                              eval_dict['gallery']['features'], gallery_data_set.labels)
        else:
            acc_list = recall(eval_dict['test']['features'], test_data_set.labels, recall_ids)
    desc = 'Test Epoch {}/{} '.format(epoch, num_epochs)
    for index, rank_id in enumerate(recall_ids):
        desc += 'R@{}:{:.2f}% '.format(rank_id, acc_list[index] * 100)
        results['test_recall@{}'.format(rank_id)].append(acc_list[index] * 100)
    print(desc)
    return acc_list[0]

if  __name__ == '__main__':
    save_name_pre = '{}_{}_{}_{}_{}_{}_{}_{}'.format(data_name, backbone_type, gd_config, feature_dim,
                                                        smoothing, temperature, margin, batch_size)

    results = {'train_loss': [], 'train_accuracy': []}
    for recall_id in recalls:
        results['test_recall@{}'.format(recall_id)] = []
    train_data_set = ImageReader(data_path, data_name, 'train')
    train_sample = MPerClassSampler(train_data_set.labels, batch_size)
    train_data_loader = DataLoader(train_data_set, batch_sampler=train_sample, num_workers=6)
    test_data_set = ImageReader(data_path, data_name, 'query' if data_name == 'isc' else 'test')
    test_data_loader = DataLoader(test_data_set, batch_size, shuffle=False, num_workers=1)
    eval_dict = {'test': {'data_loader': test_data_loader}}

    # model setup, model profile, optimizer config and loss definition
    model = Model(backbone_type, gd_config, feature_dim, num_classes=len(train_data_set.class_to_idx)).to(device)
    flops, params = profile(model, inputs=(torch.randn(1, 3, 224, 224).to(device),))
    flops, params = clever_format([flops, params])
    print('# Model Params: {} FLOPs: {}'.format(params, flops))
    optimizer = Adam(model.parameters(), lr=1e-4)
    lr_scheduler = MultiStepLR(optimizer, milestones=[int(0.6 * num_epochs), int(0.8 * num_epochs)], gamma=0.1)
    class_criterion = LabelSmoothingCrossEntropyLoss(smoothing=smoothing, temperature=temperature)
    feature_criterion = BatchHardTripletLoss(margin=margin)

    best_recall = 0.0
    print(len(train_data_set))
    for epoch in tqdm(range(1, num_epochs + 1)):
            train_loss, train_accuracy = train(model, optimizer, train_data_loader)
            results['train_loss'].append(train_loss)
            results['train_accuracy'].append(train_accuracy)
            rank = test(model, recalls)
            lr_scheduler.step()

            data_base = {}
            if rank > best_recall:
                best_recall = rank
                data_base['test_images'] = test_data_set.images
                data_base['test_labels'] = test_data_set.labels
                data_base['test_features'] = eval_dict['test']['features']
                if data_name == 'isc':
                    data_base['gallery_images'] = gallery_data_set.images
                    data_base['gallery_labels'] = gallery_data_set.labels
                    data_base['gallery_features'] = eval_dict['gallery']['features']
                torch.save(model.state_dict(), './results/{}_model.pth'.format(save_name_pre))
                print('Model saved at {}'.format(save_name_pre))
                torch.save(data_base, 'results/{}_data_base.pth'.format(save_name_pre))