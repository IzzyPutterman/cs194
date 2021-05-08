import torch
import torch.nn as nn
import numpy as np
from flask import Flask, jsonify, request
import io
from PIL import Image
import smart_open

app = Flask(__name__)

class TanhScale(nn.Module):
    def __init__(self, mean, scale):
        super().__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tanh = nn.Tanh()
        self.scale = torch.FloatTensor([scale]).to(device)
        self.mean = torch.FloatTensor([mean]).to(device)

    def forward(self, x):
        x = self.tanh(x)
        x = x * self.scale + self.mean
        return x
device = torch.device("cpu")
model_temp = torch.load("api_server/model_temp_29_sfsea_mod.pt", map_location=torch.device('cpu'))
model_rain = torch.load("api_server/model_rain_83_sfsea.pt", map_location=torch.device('cpu'))
tanhscale  = TanhScale(40, 55)
def forward_temp(img):
    img = torch.as_tensor(img).to(device).float()
    inter = model_temp(img)
    return tanhscale(inter)

def forward_rain(img):
    img = torch.as_tensor(img).to(device).float()
    return model_rain(img)

def predicts(img):
    # print(img)
    img = preprocess(img)
    model_temp.eval()
    model_rain.eval()
    temps = forward_temp(img).detach().numpy()[0]
    rains = forward_rain(img).detach().numpy()[0]
    return temps[0], temps[1], rains

def preprocess(img):
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    if len(img.shape) == 3:
      img = img[None]
    if img.max() > 1:
      img = img/255.
    if img.shape[1] != 3:
      img = img.transpose(0, 3, 1, 2)
    for i in range(img.shape[1]):
      img[:, i, :, :] = (img[:, i, :, :] - mean[i]) / std[i]
    return img


@app.route('/predict', methods=['GET','POST'])
def predict():
    # print(request.method)
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        img = np.array(Image.open(io.BytesIO(img_bytes)))
        low, high, rain = predicts(img)
        if rain[0] > rain[1]:
            return jsonify({'low': str(low), 'high': str(high), 'rain': False})
        else:
            return jsonify({'low': str(low), 'high': str(high), 'rain': True})
    if request.method == "GET":
        image_url = request.args.get("image_url")
        # print(image_url)
        if image_url is None:
            return "no image_url defined in query string"
        img = np.array(read_image_pil(image_url))
        # print(img.shape)
        low, high, rain = predicts(img)
        if rain[0] > rain[1]:
            return jsonify({'low': str(low), 'high': str(high), 'rain': False})
        else:
            return jsonify({'low': str(low), 'high': str(high), 'rain': True})


def read_image_pil(image_uri):
    with smart_open.open(image_uri, "rb") as image_file:
        return read_image_pil_file(image_file)


def read_image_pil_file(image_file):
    with Image.open(image_file) as image:
        image = image.convert(mode=image.mode)
        return image

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=False)
