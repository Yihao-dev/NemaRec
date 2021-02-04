#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import io
import json
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request
from flask_cors import CORS
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = '/Users/wangyihao/Desktop/z/' #我们上线的时候用server

@app.route('/upload',methods=['POST'])
def upload():
    file = request.files['file'];
#     content = file.read() if binary works use this
    filepath = UPLOAD_FOLDER+file.filename
    file.save(filepath)
    file.close()

    #如果不可以totensor，在此读路径
    return {'success': False, 'data': predict(filepath),'mag': 'hahah'}

def transform_image(image_bytes):
    
    transform = transforms.Compose([transforms.Resize((224,224)),
                                    transforms.Grayscale(num_output_channels=3),
                                    transforms.ToTensor()])
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)


def get_model():
    global model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print(device)

    model=models.resnet101(pretrained=False)
    model.fc = nn.Linear(2048, 19)
    pthfile='./Desktop/resnet101_data_expansion_3x_best_model_parameters.pt'
    model_PT=torch.load(pthfile,map_location=device)
    model.load_state_dict(model_PT["model_state_dict"])
    model.eval()#necessary to add brackets
    print('the model is loaded')
get_model()
   
#class_mapping = ('Helicotylenchus','Xenocriconema','Mylonchulus','Ditylenchus','Panagrolaimus','Rhbiditis','Pratylenchus','Acrobeloides','Pristionchus','Aphelenchoides','Axonchium','Aporcelaimus','Discolimus','Eudorylaimus','Mesodorylaimus','Miconchus','Dorylaimus','Amplimerlinius','Acrobeles')
class_mapping=('Acrobeles','Acrobeloides','Amplimerlinius','Aphelenchoides','Aporcelaimus','Axonchium','Discolaimus','Ditylenchus','Dorylaimus','Eudorylaimus','Helicotylenchus','Mesodorylaimus','Miconchus','Mylonchulus','Panagrolaimus','Pratylenchus','Pristionchus','Rhbiditis','Xenocriconema')

def get_category(image):
  # read the image in binary form
#     with open(image, 'rb') as file:
#         image_bytes = file.read()
    
    file = open(image,'rb')
    image_bytes = file.read()
    transformed_image = transform_image(image_bytes=image_bytes)
    # use the model to predict the class
#     print(type(transformed_image))
#     #print(type(images))
#     print(transformed_image.size())
#     print(transformed_image)
    outputs = model(transformed_image)
    _, category = torch.max(outputs,1)
    # return the value
    predicted_idx = category.item()
    #print(class_mapping[predicted_idx])
    return class_mapping[predicted_idx]

@app.route('/predict', methods=['POST'])
def predict(file):
    class_name = get_category(image=file)
    return class_name



if __name__ == '__main__':
    app.run()

