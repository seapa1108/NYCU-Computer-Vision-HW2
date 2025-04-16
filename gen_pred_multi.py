import os
import json
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import torchvision.transforms.functional as F
from ensemble_boxes import weighted_boxes_fusion
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
)
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import FasterRCNN


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


default_transform = ComposeDetection([ToTensorDetection()])
norm_transform = ComposeDetection(
    [
        ToTensorDetection(),
        NormalizeDetection(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)


data_dir = "nycu-hw2-data"
test_dir = os.path.join(data_dir, "test")


def build_model(model_weight_path, device, variant="resnet50_v2"):
    num_classes = 11  # background + 10 digit classes

    if variant == "resnet50_v2":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        )
    elif variant == "mobilenet_v3":
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
            weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        )
    elif variant == "resnet101_custom":
        backbone = resnet_fpn_backbone(
            backbone_name="resnet101",
            weights=torchvision.models.ResNet101_Weights.DEFAULT,
        )
        model = FasterRCNN(backbone, num_classes=num_classes)
        model.to(device)
        model.eval()
        state_dict = torch.load(model_weight_path, map_location=device)
        model.load_state_dict(state_dict)
        return model
    else:
        raise ValueError("Unknown model variant.")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    state_dict = torch.load(model_weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def combine_wbf_detections(
    all_model_boxes,
    all_model_scores,
    all_model_labels,
    image_width,
    image_height,
    iou_thr=0.5,
    skip_box_thr=0.85,
):
    normalized_boxes = [
        [
            [
                x / image_width,
                y / image_height,
                (x + w) / image_width,
                (y + h) / image_height,
            ]
            for (x, y, w, h) in boxes
        ]
        for boxes in all_model_boxes
    ]

    boxes, scores, labels = weighted_boxes_fusion(
        normalized_boxes,
        all_model_scores,
        all_model_labels,
        iou_thr=iou_thr,
        skip_box_thr=skip_box_thr,
    )

    fused_dets = []
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        x1 *= image_width
        y1 *= image_height
        x2 *= image_width
        y2 *= image_height
        fused_dets.append(
            {
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(score),
                "category_id": int(label),
            }
        )
    return fused_dets


def main():
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print("Using device:", device)

    model_info = [
        {
            "path": "model/best_model3.pth",
            "variant": "resnet50_v2",
            "transform": default_transform,
        },
        {
            "path": "model/faster_rcnn_digit.pth",
            "variant": "mobilenet_v3",
            "transform": default_transform,
        },
        {
            "path": "model/best_model2.pth",
            "variant": "resnet101_custom",
            "transform": default_transform,
        },
        {
            "path": "model/best_model.pth",
            "variant": "resnet50_v2",
            "transform": norm_transform,
        },
    ]

    ensemble = []
    for info in model_info:
        try:
            model = build_model(info["path"], device, variant=info["variant"])
            ensemble.append((model, info["transform"]))
            print(f"Successfully loaded model: {info['path']}")
        except Exception as e:
            print(f"Failed to load model {info['path']}: {str(e)}")

    detection_threshold = 0.85

    test_images = [f for f in os.listdir(test_dir) if f.endswith(".png")]
    detection_results = []

    print("Starting Inference on Test Set...")
    for image_file in tqdm(test_images, desc="Inference"):
        image_path = os.path.join(test_dir, image_file)
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        model_boxes = []
        model_scores = []
        model_labels = []

        for idx, (model, model_transform) in enumerate(ensemble):
            try:
                image_tensor, _ = model_transform(image, None)
                image_tensor = image_tensor.to(device)

                with torch.no_grad():
                    output = model([image_tensor])[0]

                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()

                converted_boxes = []
                filtered_scores = []
                filtered_labels = []

                for box, score, label in zip(boxes, scores, labels):
                    if score < detection_threshold:
                        continue
                    x1, y1, x2, y2 = box
                    converted_boxes.append([x1, y1, x2 - x1, y2 - y1])
                    filtered_scores.append(score)
                    filtered_labels.append(label)

                model_boxes.append(converted_boxes)
                model_scores.append(filtered_scores)
                model_labels.append(filtered_labels)

            except Exception as e:
                print(
                    f"Error with model {idx} on image {image_file}: {str(e)}"
                )

        if not any(len(boxes) > 0 for boxes in model_boxes):
            continue

        fused = combine_wbf_detections(
            model_boxes,
            model_scores,
            model_labels,
            image_width=width,
            image_height=height,
            iou_thr=0.35,
            skip_box_thr=0.85,
        )

        base_id = os.path.splitext(image_file)[0]
        try:
            image_id = int(base_id)
        except ValueError:
            image_id = base_id

        for det in fused:
            det["image_id"] = image_id
            detection_results.append(det)

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


if __name__ == "__main__":
    main()
