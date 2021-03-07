from flask import Flask,render_template, request, flash, request, redirect, url_for, send_from_directory
import os

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
num_retrieval = 50

model = Model(backbone_type, gd_config, feature_dim, num_classes=22)
flops, params = profile(model, inputs=(torch.randn(1, 3, 224, 224),))
model.apply(set_bn_eval)
model.load_state_dict(torch.load('./results/vnfood_model_final.pth', map_location=torch.device('cpu')))
model.eval()

global features
features = torch.load('./toolboxs/features.pth', map_location=torch.device('cpu'))


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])

transforms = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

app = Flask(__name__)
app.secret_key = 'super secret key'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

from werkzeug.utils import secure_filename

global IMAGE_QUERY_OFFLINE_NAME
global IMAGE_QUERY_OFFLINE_PATH
global IMAGE_QUERY_DATABASE
IMAGE_QUERY_OFFLINE_NAME = 'bubu.png'
IMAGE_QUERY_OFFLINE_PATH = './dataset/vn_food/test'
IMAGE_QUERY_DATABASE = './static/test'
save_path = './users'
INPUT_FILE_NAME = "input.jpg"

# @app.route("/homepage")
# def homepage():
#     return render_template('homepage.html')

@app.after_request
def add_header(response):
    # response.cache_control.no_store = Tru
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    global image_ids
    # If users post an image onto the server
    print(request)
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        print(file.filename)
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        # if user submit a file and that file is valid
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # construct the input path
            input_file_path = os.path.join(save_path, INPUT_FILE_NAME)
            #  save that file to the folder
            file.save(input_file_path)
        query_url = input_file_path
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
        result_imgs = []
        for i in range(num_retrieval):  
            img = os.path.join(IMAGE_QUERY_DATABASE, os.listdir(IMAGE_QUERY_DATABASE)[idx[i]])
            result_imgs.append(img)
        return render_template('all_images.html', links = result_imgs)
    else:
        return render_template('homepage.html')

@app.route('/sample/<idx>/')
def sample(idx):
    if idx == '0':
        path = './static/images/'
        imgs = []
        for img in os.listdir(path):
            imgs.append(img)
        return render_template('searchBySample.html', imgs=imgs)
    else:
        img_name = idx
        # print(IMAGE_QUERY_OFFLINE_NAME)
        return redirect(url_for('sampleResult', messages = img_name))

@app.route('/sampleResult')
def sampleResult():
    IMAGE_QUERY_OFFLINE_NAME = request.args['messages']
    query_url = '{}/{}'.format(IMAGE_QUERY_OFFLINE_PATH, IMAGE_QUERY_OFFLINE_NAME)
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
    result_imgs = []
    for i in range(num_retrieval):  
        img = os.path.join(IMAGE_QUERY_DATABASE, os.listdir(IMAGE_QUERY_DATABASE)[idx[i]])
        result_imgs.append(img)
    return render_template('all_images.html', links = result_imgs)
    
        

if __name__ == '__main__':
    app.run(port='80', debug=True)

    