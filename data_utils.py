import argparse
import os

import torch
from PIL import Image
from scipy.io import loadmat
from tqdm import tqdm


def read_txt(path, data_num):
    data = {}
    for line in open(path, 'r', encoding='utf-8'):
        if data_num == 2:
            data_1, data_2 = line.split()
        else:
            data_1, data_2, data_3, data_4, data_5 = line.split()
            data_2 = [data_2, data_3, data_4, data_5]
        data[data_1] = data_2
    return data

def carDataProcessing(data_path, folder_name, data_type, _dict):
    ALLOWED_LABELS = ['1', '2', '3', '4', '5', '6']
    if folder_name == 'train':
	    annotations = loadmat('{}/car/{}/cars_train_annos.mat'.format(data_path, folder_name))['annotations'][0]
    else:
	    annotations = loadmat('{}/car/{}/cars_test_annos_withlabels.mat'.format(data_path, folder_name))['annotations'][0]
    for img in tqdm(annotations, desc='process {} data for car dataset'.format(data_type)):
        img_name, img_label = str(img[-1][0]), str(img[4][0][0])
        if img_label in ALLOWED_LABELS:
            if data_type == 'uncropped':
                img = Image.open('{}/car/{}/{}'.format(data_path, folder_name, img_name)).convert('RGB')
            else:
                x1, y1, x2, y2 = int(img[0][0][0]), int(img[1][0][0]), int(img[2][0][0]), int(img[3][0][0])
                img = Image.open('{}/car/{}/{}'.format(data_path, folder_name, img_name)).convert('RGB').crop((x1, y1, x2, y2))
            save_name = '{}/car/{}/{}/{}'.format(data_path, folder_name, data_type, os.path.basename(img_name))
            img.save(save_name)
            if img_label in _dict:
                _dict[img_label].append(save_name)
            else:
                _dict[img_label] = [save_name]
    torch.save({'{}'.format(folder_name): _dict}, '{}/car/{}/{}_data_dicts.pth'.format(data_path, folder_name, data_type))

def process_car_data(data_path, data_type):
    folder_types = ['train', 'test']
    for folder_name in folder_types:
        if not os.path.exists('{}/car/{}/{}'.format(data_path, folder_name, data_type)):
            os.mkdir('{}/car/{}/{}'.format(data_path, folder_name, data_type))
        _dict = {}
        carDataProcessing(data_path, folder_name, data_type, _dict)
    


def foodDataProcessing(data_path):
    data_name = 'vn_food'
    folders = ['train', 'test']
    dict_name = 'data_dicts.pth'
    for folder in folders:
        new_dicts = {}
        temp_path = '{}/{}/{}'.format(data_path, data_name, folder)
        data_dicts = torch.load('{}/data_dicts.pth'.format(temp_path))
        for key in data_dicts.keys():
            new_dicts[key] = []
            img_names = data_dicts[key]
            for img_name in img_names:
                new_name = '{}/{}/{}'.format(data_path, data_name, folder) + img_name
                new_dicts[key].append(new_name)
        torch.save(new_dicts, '{}/{}/{}/{}'.format(data_path, data_name, folder, dict_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process datasets')
    parser.add_argument('--data_path', default='/home/data', type=str, help='datasets path')

    opt = parser.parse_args()

    process_car_data('{}'.format(opt.data_path), 'uncropped')
