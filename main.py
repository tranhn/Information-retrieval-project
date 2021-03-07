import torch
import numpy as np
import os
from PIL import Image

from thop import profile, clever_format
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from model import Model, set_bn_eval
from my_utils import recall, LabelSmoothingCrossEntropyLoss, BatchHardTripletLoss, ImageReader, MPerClassSampler

data_path = './dataset'
data_name = 'vn_food'
batch_size = 16
backbone_type = 'wide_resnet50_2'
gd_config = 'SG'
feature_dim = 512
num_epochs = 35
smoothing = 0.1
temperature = 0.5
margin = 0.1
recalls = 1,2,4,8

model = Model(backbone_type, gd_config, feature_dim, num_classes=22)
flops, params = profile(model, inputs=(torch.randn(1, 3, 224, 224),))
model.apply(set_bn_eval)
model.load_state_dict(torch.load('./results/vnfood_model_final.pth', map_location=torch.device('cpu')))
model.eval()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])

transforms = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])


if __name__ == '__main__':
    
    query_url = './dataset/vn_food/database/203.png'
    num_retrieval = 10
    features = torch.load('./toolboxs/features.pth', map_location=torch.device('cpu'))
    print(features.size())
    database_path = './dataset/vn_food/test'
    with torch.no_grad():
        img = Image.open(query_url).convert('RGB')
        img = transforms(img)
        img = (img.unsqueeze(0))
        query = model(img)[0]
        query = query.flatten(start_dim=1)
    #for cosine metric
    cosine = torch.nn.CosineSimilarity(1)
    results = cosine(query, features)
    _, idx = results.topk(num_retrieval)
    for i in range(num_retrieval):  
        img = Image.open(os.path.join(database_path, os.listdir(database_path)[idx[i]]))
        img.show()
        print(os.path.join(database_path, os.listdir(database_path)[idx[i]]))
    #   plt.imshow(img)
    #   plt.show()