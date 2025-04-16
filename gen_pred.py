import os
import json
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
from PIL import Image
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import torchvision.transforms.functional as F


class ToTensorDetection:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class ComposeDetection:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


transform = ComposeDetection([ToTensorDetection()])

data_dir = "nycu-hw2-data"
test_dir = os.path.join(data_dir, "test")

device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)
print("Using device:", device)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
    weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
)

num_classes = 11
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

model_path = "best_model.pth"
model.load_state_dict(
    torch.load(model_path, map_location=device, weights_only=True)
)
model.to(device)
model.eval()

test_images = [f for f in os.listdir(test_dir) if f.endswith(".png")]
detection_results = []
detection_threshold = 0.7

print("Starting Inference on Test Set...")
for image_file in tqdm(test_images, desc="Inference"):
    image_path = os.path.join(test_dir, image_file)
    image = Image.open(image_path).convert("RGB")

    image_tensor, _ = transform(image, None)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model([image_tensor])

    output = outputs[0]
    boxes = output["boxes"].cpu().numpy()
    labels = output["labels"].cpu().numpy()
    scores = output["scores"].cpu().numpy()

    converted_boxes = []
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        converted_boxes.append(
            [
                float(x_min),
                float(y_min),
                float(x_max - x_min),
                float(y_max - y_min),
            ]
        )

    base_id = os.path.splitext(image_file)[0]
    try:
        image_id = int(base_id)
    except ValueError:
        image_id = base_id

    for bbox, label, score in zip(converted_boxes, labels, scores):
        if score < detection_threshold:
            continue
        detection_results.append(
            {
                "image_id": image_id,
                "bbox": bbox,
                "score": float(score),
                "category_id": int(label),
            }
        )

with open("pred.json", "w") as f:
    json.dump(detection_results, f, indent=4)
print("pred.json generated.")

detections_by_image = defaultdict(list)
for det in detection_results:
    detections_by_image[det["image_id"]].append(det)


def map_category_to_digit(category_id):
    return str(category_id - 1)


results = []
for image_file in test_images:
    base_id = os.path.splitext(image_file)[0]
    try:
        image_key = int(base_id)
    except ValueError:
        image_key = base_id

    image_detections = detections_by_image.get(image_key, [])
    if len(image_detections) == 0:
        pred_label = -1
    else:
        image_detections.sort(key=lambda det: det["bbox"][0])
        digits = [
            map_category_to_digit(det["category_id"])
            for det in image_detections
        ]
        pred_label = "".join(digits)

    results.append([image_key, pred_label])

df = pd.DataFrame(results, columns=["image_id", "pred_label"])
df.to_csv("pred.csv", index=False)
print("pred.csv generated.")
