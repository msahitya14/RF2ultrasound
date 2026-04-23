import argparse, io
from flask import Flask, request, jsonify
from PIL import Image
import torch

from model import UltrasoundLocalizer
from dataset import denormalize_x, denormalize_y
from predict import load_model, get_transform

app = Flask(__name__)
model     = None
transform = None
device    = None

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image field'}), 400
    file  = request.files['image']
    img   = Image.open(io.BytesIO(file.read())).convert('RGB')
    inp   = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(inp).squeeze(0).cpu()
    x = denormalize_x(pred[0:1]).item()
    y = denormalize_y(pred[1:2]).item()
    return jsonify({'x': x, 'y': y})

# Also expose the existing /angles endpoint if needed
@app.route('/angles', methods=['GET'])
def angles():
    return jsonify({'x': 0.0, 'y': 0.0,
                    'calibratedAt': 'N/A', 'updatedAt': 'N/A'})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--port', type=int, default=5001)
    args = parser.parse_args()

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model     = load_model(args.checkpoint, device)
    transform = get_transform(224)
    app.run(host='0.0.0.0', port=args.port)
