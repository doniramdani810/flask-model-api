import os
import base64
import io
import logging
import uuid
import random
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import torch
import pickle
import cv2
from PIL import Image
from difflib import SequenceMatcher
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from torchvision.ops import nms

# —————— Configuration ——————
MODEL_A_WEIGHTS = os.getenv("MODEL_A_WEIGHTS", "best_model.pth")
MODEL_A_MAPPING = os.getenv("MODEL_A_MAPPING", "class_mapping.pkl")
MODEL_B_WEIGHTS = os.getenv("MODEL_B_WEIGHTS", "best_model1.pth")
MODEL_B_MAPPING = os.getenv("MODEL_B_MAPPING", "class_mapping1.pkl")
INPUT_SIZE_A = (int(os.getenv("INPUT_A_W", 720)), int(os.getenv("INPUT_A_H", 120)))
INPUT_SIZE_B = (int(os.getenv("INPUT_B_W", 352)), int(os.getenv("INPUT_B_H", 198)))
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", 0.2))
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", 0.1))
SIMILARITY_CUTOFF = float(os.getenv("SIM_CUTOFF", 0.6))
OVERLAP_REMOVAL_THRESHOLD = float(os.getenv("OVERLAP_REMOVAL", 30))
SAVE_DEBUG_IMAGES = True
DEBUG_IMAGE_DIR = "debug_images"

# —————— Logging ——————
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# —————— Flask App Entrypoint ——————
app = Flask(__name__)
CORS(app)

# —————— Helpers ——————

def get_similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def add_black_background_no_scale(img, tw, th):
    oh, ow = img.shape[:2]
    if ow > tw or oh > th:
        x0, y0 = (ow - tw)//2, (oh - th)//2
        cropped = img[y0:y0+th, x0:x0+tw]
        pad_x = pad_y = 0
        crop_x, crop_y = x0, y0
        canvas = cropped
    else:
        pad_x, pad_y = (tw - ow)//2, (th - oh)//2
        crop_x = crop_y = 0
        canvas = np.zeros((th, tw, 3), dtype=np.uint8)
        canvas[pad_y:pad_y+oh, pad_x:pad_x+ow] = img
    return canvas, pad_x, pad_y, crop_x, crop_y


def remove_overlaps(dets, thresh):
    filtered = []
    for d in dets:
        if not any(np.hypot(d['center'][0]-f['center'][0], d['center'][1]-f['center'][1])<thresh for f in filtered):
            filtered.append(d)
    return filtered


def save_debug_images_paired(img_hint, img_main):
    os.makedirs(DEBUG_IMAGE_DIR, exist_ok=True)
    uid = uuid.uuid4().hex[:8]
    hint_path = os.path.join(DEBUG_IMAGE_DIR, f"hint_{uid}.jpg")
    main_path = os.path.join(DEBUG_IMAGE_DIR, f"main_{uid}.jpg")
    cv2.imwrite(hint_path, img_hint)
    cv2.imwrite(main_path, img_main)
    logger.info(f"Saved debug images: {hint_path}, {main_path}")


def decode_image(b64_string):
    if ',' in b64_string:
        b64_string = b64_string.split(',')[1]
    data = base64.b64decode(b64_string)
    img = Image.open(io.BytesIO(data)).convert('RGB')
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# —————— Model Loading ——————

def load_model(weights_path, mapping_path):
    logger.info(f"Loading model {weights_path}")
    with open(mapping_path,'rb') as f:
        mp = pickle.load(f)
    idx2cls = mp['idx_to_class']
    num_classes = len(mp['classes']) + 1
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(dev).eval()
    return model, idx2cls, dev

modelA, clsA, deviceA = load_model(MODEL_A_WEIGHTS, MODEL_A_MAPPING)
modelB, clsB, deviceB = load_model(MODEL_B_WEIGHTS, MODEL_B_MAPPING)

# —————— Prediction ——————

def predict_single(model, idx2cls, device, img, target_size):
    tw, th = target_size
    canvas, pad_x, pad_y, crop_x, crop_y = add_black_background_no_scale(img, tw, th)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    tensor = transforms.ToTensor()(rgb).unsqueeze(0).to(device)
    with torch.no_grad(): out = model(tensor)[0]
    boxes = out['boxes'].cpu().numpy()
    scores = out['scores'].cpu().numpy()
    labels = out['labels'].cpu().numpy()
    keep = nms(torch.tensor(boxes), torch.tensor(scores), IOU_THRESHOLD).numpy()
    dets = []
    for i in keep:
        if scores[i] < CONF_THRESHOLD:
            continue
        x1,y1,x2,y2 = boxes[i]
        cx, cy = (x1+x2)/2, (y1+y2)/2
        orig_x = cx - pad_x + crop_x
        orig_y = cy - pad_y + crop_y
        if not (0 <= orig_x < img.shape[1] and 0 <= orig_y < img.shape[0]):
            continue
        cls_name = idx2cls[int(labels[i])]
        dets.append({'class': cls_name, 'center': (orig_x, orig_y), 'score': float(scores[i])})
    return remove_overlaps(dets, OVERLAP_REMOVAL_THRESHOLD)

# —————— Routes ——————

@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.json or {}
    hint_b64 = data.get('hint_image')
    main_b64 = data.get('main_image')
    if not hint_b64 or not main_b64:
        return jsonify({'error': 'Missing images'}), 400

    img_hint = decode_image(hint_b64)
    img_main = decode_image(main_b64)

    if SAVE_DEBUG_IMAGES:
        save_debug_images_paired(img_hint, img_main)

    detsA = predict_single(modelA, clsA, deviceA, img_hint, INPUT_SIZE_A)
    detsB = predict_single(modelB, clsB, deviceB, img_main, INPUT_SIZE_B)

    detsA = sorted(detsA, key=lambda d: d['center'][0])
    labelsA = [d['class'] for d in detsA]
    used_b = set()
    ordered = []

    # 1) Map A -> B by similarity
    for label in labelsA:
        best_idx, best_score = None, 0.0
        for j, b in enumerate(detsB):
            if j in used_b: continue
            score = get_similarity(label, b['class'])
            if score > best_score:
                best_score, best_idx = score, j
        if best_idx is not None and best_score >= SIMILARITY_CUTOFF:
            ordered.append(detsB[best_idx])
            used_b.add(best_idx)
        else:
            ordered.append(None)

    # 2) Fill None with random leftovers
    leftovers = [d for idx, d in enumerate(detsB) if idx not in used_b]
    random.shuffle(leftovers)
    li = 0
    for i in range(len(ordered)):
        if ordered[i] is None and li < len(leftovers):
            ordered[i] = leftovers[li]
            li += 1

    # 3) Trim to length of detsA
    ordered = ordered[:len(detsA)]

    coords = [{'x': round(d['center'][0], 2), 'y': round(d['center'][1], 2)} for d in ordered]
    return jsonify({'coordinates': coords})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200
