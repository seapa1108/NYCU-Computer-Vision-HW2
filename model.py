import os
import json
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class ToTensorDetection:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class NormalizeDetection:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class ComposeDetection:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


transform = ComposeDetection(
    [
        ToTensorDetection(),
        NormalizeDetection(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

data_dir = "nycu-hw2-data"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "valid")
test_dir = os.path.join(data_dir, "test")

train_ann_file = os.path.join(data_dir, "train.json")
val_ann_file = os.path.join(data_dir, "valid.json")

train_dataset = CocoDetection(
    root=train_dir, annFile=train_ann_file, transforms=transform
)
val_dataset = CocoDetection(
    root=val_dir, annFile=val_ann_file, transforms=transform
)


def collate_fn(batch):
    return tuple(zip(*batch))


train_loader = DataLoader(
    train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn
)
val_loader = DataLoader(
    val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn
)

device = (
    torch.device("cuda:1")
    if torch.cuda.is_available()
    else torch.device("cpu")
)
print("Using device:", device)

num_classes = 11  # 10 digits + background

model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
    weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
)


in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

model.to(device)

optimizer = torch.optim.SGD(
    [p for p in model.parameters() if p.requires_grad],
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005,
)

lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=4, gamma=0.35
)


def generate_coco_predictions(model, loader, device, detection_threshold=0.5):
    model.eval()
    pred_results = []
    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            for idx, output in enumerate(outputs):
                if len(targets[idx]) > 0 and "image_id" in targets[idx][0]:
                    image_id = targets[idx][0]["image_id"]
                else:
                    image_id = idx
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
                for bbox, label, score in zip(converted_boxes, labels, scores):
                    if score < detection_threshold:
                        continue
                    pred_results.append(
                        {
                            "image_id": image_id,
                            "bbox": bbox,
                            "score": float(score),
                            "category_id": int(label),
                        }
                    )
    return pred_results


def evaluate_model_mAP(
    model, loader, device, gt_ann_file, detection_threshold=0.5
):
    pred_results = generate_coco_predictions(
        model, loader, device, detection_threshold=detection_threshold
    )
    temp_pred_file = "temp_pred.json"
    with open(temp_pred_file, "w") as f:
        json.dump(pred_results, f)
    coco_gt = COCO(gt_ann_file)
    coco_dt = coco_gt.loadRes(temp_pred_file)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    mAP = coco_eval.stats[0]
    return mAP


num_epochs = 10
best_mAP = 0

train_losses = []
val_losses = []
val_mAPs = []

print("Starting Training...")
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    progress_bar = tqdm(
        train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False
    )
    for images, targets in progress_bar:
        images = [img.to(device) for img in images]
        reformatted_targets = []
        for target in targets:
            boxes = []
            labels = []
            for ann in target:
                x_min, y_min, w, h = ann["bbox"]
                boxes.append([x_min, y_min, x_min + w, y_min + h])
                labels.append(ann["category_id"])
            reformatted_targets.append(
                {
                    "boxes": torch.as_tensor(boxes, dtype=torch.float32).to(
                        device
                    ),
                    "labels": torch.as_tensor(labels, dtype=torch.int64).to(
                        device
                    ),
                }
            )
        loss_dict = model(images, reformatted_targets)
        losses = sum(loss for loss in loss_dict.values())
        current_loss = losses.item()
        epoch_loss += current_loss

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        progress_bar.set_postfix(loss=f"{current_loss:.4f}")

    avg_train_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}")

    lr_scheduler.step()

    model.train()
    total_val_loss = 0.0
    count = 0
    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            reformatted_targets = []
            for target in targets:
                boxes = []
                labels = []
                for ann in target:
                    x_min, y_min, w, h = ann["bbox"]
                    boxes.append([x_min, y_min, x_min + w, y_min + h])
                    labels.append(ann["category_id"])
                reformatted_targets.append(
                    {
                        "boxes": torch.as_tensor(
                            boxes, dtype=torch.float32
                        ).to(device),
                        "labels": torch.as_tensor(
                            labels, dtype=torch.int64
                        ).to(device),
                    }
                )
            loss_dict = model(images, reformatted_targets)
            losses = sum(loss for loss in loss_dict.values())
            total_val_loss += losses.item()
            count += 1
    avg_val_loss = total_val_loss / count
    val_losses.append(avg_val_loss)
    print(f"Epoch {epoch+1} - Val Loss: {avg_val_loss:.4f}")

    mAP = evaluate_model_mAP(
        model, val_loader, device, val_ann_file, detection_threshold=0.5
    )
    val_mAPs.append(mAP)
    print(f"Epoch {epoch+1} - Val mAP: {mAP:.3f}")

    fig = plt.figure(figsize=(8, 6))

    plt.subplot(2, 1, 1)
    plt.plot(
        range(1, epoch + 2),
        train_losses,
        label="Train Loss",
        linestyle="-",
        marker="o",
    )
    plt.plot(
        range(1, epoch + 2),
        val_losses,
        label="Val Loss",
        linestyle="--",
        marker="o",
    )
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(
        range(1, epoch + 2),
        val_mAPs,
        label="Val mAP",
        linestyle="-",
        marker="o",
        color="green",
    )
    plt.xlabel("Epochs")
    plt.ylabel("mAP")
    plt.title("Validation mAP")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("hihi.png")
    plt.close(fig)

    if best_mAP < mAP:
        best_mAP = mAP
        torch.save(model.state_dict(), "best_model.pth")
        print(f"Epoch {epoch+1}: New best map - Model saved.")
    else:
        print(f"Epoch {epoch+1}: No improvement.")


print("Training completed and best model saved.")

model.eval()
test_images = [f for f in os.listdir(test_dir) if f.endswith(".png")]
detection_results = []
detection_threshold = 0.7

print("Starting Inference on Test Set...")
for image_file in tqdm(test_images, desc="Test Inference"):
    image_path = os.path.join(test_dir, image_file)
    image = Image.open(image_path).convert("RGB")
    # For inference, pass target as None.
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
