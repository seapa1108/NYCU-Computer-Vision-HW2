## NYCU Computer Vision 2025 Spring HW2
- Student ID: 111550084
- Name: 林辰恩

### Introduction
This project focuses on digit recognition from images using a two-stage approach. The first stage predicts bounding boxes and classifies each digit, while the second stage combines the detected digits to form the final numeric prediction. To achieve this, I use a pre-trained Faster R-CNN model with a ResNet-50 backbone, which is fine-tuned to support 11-class classification.

### How to install 👹
```bash
git clone https://github.com/seapa1108/NYCU-Computer-Vision-HW2
cd NYCU-Computer-Vision-HW2
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install pandas tqdm matplotlib pillow pycocotools
```

### Performance Snapshot
<p align="center">
  <img src="./image/hihihi.png">
</p>