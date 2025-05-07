# Text Line Segmentation with YOLO11

This repository contains scripts for training and using YOLO11 models for text line segmentation in historical documents.

## Scripts Overview

### 1. `convert_page_to_yolo.py`
Converts PAGE-XML annotations to YOLO format for segmentation training.

```bash
python convert_page_to_yolo.py input_dir output_dir --target-height 640 --element-type textline
```

### 2. `visualize_masks.py`
Visualizes YOLO segmentation masks on images.

```bash
python visualize_masks.py --dataset /path/to/dataset --output-dir /path/to/output
```

### 3. `train.py`
Basic training script for YOLO11 segmentation models.

```bash
python train.py \
    --dataset /path/to/dataset \
    --model-size m \
    --batch-size 8 \
    --epochs 100 \
    --pretrained \
    --val \
    --plots
```
Metrics:
![image](https://github.com/user-attachments/assets/4372c43e-3495-493c-91c4-3fc77fb42a2e)
### 4. `train_improved.py`
Enhanced training script with improved augmentation and training parameters.

```bash
python train_improved.py \
    --dataset /path/to/dataset \
    --model-size m \
    --batch-size 12 \
    --epochs 100 \
    --pretrained \
    --val \
    --plots
```

Key improvements in `train_improved.py`:
- Enhanced augmentation (mosaic, mixup, copy-paste)
- Better learning rate scheduling
- Improved regularization
- Optimized for segmentation performance

Metrics:
![image](https://github.com/user-attachments/assets/93bcc69c-847f-4496-b20b-f2b3f240a4cb)
### 5. `app.py`
Interactive Gradio web interface for model inference.

```bash
python app.py
```

Features:
- Lists all available model checkpoints from `runs/train/`
- Upload images for prediction
- Toggle between mask and bounding box visualization
- Adjust confidence threshold
- Real-time visualization

## Dataset Structure

The dataset should be organized as follows:
```
dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── dataset.yaml
```

The `dataset.yaml` file should contain:
```yaml
path: /path/to/dataset
train: images/train
val: images/val
names:
  0: textline
```

## Training Progress Metrics
- Box Loss: Detection accuracy
- Mask Loss: Segmentation quality
- Precision: Accuracy of detections
- Recall: Coverage of text lines
- mAP50: Mean Average Precision at 50% IoU
- mAP50-95: Mean Average Precision at various IoU thresholds

## Visualization Features
- Green masks for text lines
- Red bounding boxes (optional)
- Confidence scores
- Interactive web interface

## Important Notes
- The model is trained for single-class text line segmentation
- Supports various YOLO11 model sizes (n, s, m, l, x)
- Automatic mixed precision training is enabled
- Cosine learning rate scheduling is used
- Data augmentation is optimized for document images

## Hardware Requirements
- NVIDIA GPU with at least 12GB VRAM recommended
- Batch size should be adjusted based on available GPU memory
- For RTX 3060 12GB, recommended batch size is 8-12 for YOLO11m

## Model Performance
The model achieves high accuracy in text line segmentation with:
- High precision and recall
- Accurate mask boundaries
- Good handling of various text line orientations
- Robust performance on different document styles

## Notes

- The conversion script preserves original polygon shapes without padding
- Training uses single-class segmentation for text lines
- The model supports various sizes (nano to xlarge) for different performance requirements 

## Training Approaches Comparison

We compared different training configurations to find the optimal setup for text line segmentation. Here are the results:

### 1. Original Training (YOLO11m)
- Configuration:
  - Model: YOLO11m
  - Batch size: 8
  - Optimizer: AdamW
- Final metrics:
  - Box Loss: 0.713
  - Seg Loss: 2.057
  - mAP50(B): 0.992
  - mAP50(M): 0.912
- Training time: ~3751 seconds

### 2. Improved Training (YOLO11m)
- Configuration:
  - Model: YOLO11m
  - Batch size: 12
  - Optimizer: AdamW
  - Enhanced augmentation
- Final metrics:
  - Box Loss: 0.314
  - Seg Loss: 0.915
  - mAP50(B): 0.989
  - mAP50(M): 0.911
- Training time: ~14468 seconds

### 3. Small Model with AdamW (YOLO11s)
- Configuration:
  - Model: YOLO11s
  - Batch size: 12
  - Optimizer: AdamW
  - Enhanced augmentation
- Final metrics:
  - Box Loss: 0.291
  - Seg Loss: 0.891
  - mAP50(B): 0.991
  - mAP50(M): 0.913
- Training time: ~12000 seconds

### 4. Small Model with SGD (YOLO11s)
- Configuration:
  - Model: YOLO11s
  - Batch size: 12
  - Optimizer: SGD
  - Enhanced augmentation
- Final metrics:
  - Box Loss: 0.285
  - Seg Loss: 0.887
  - mAP50(B): 0.992
  - mAP50(M): 0.914
- Training time: ~11000 seconds

### Key Findings:
1. **Model Size**: YOLO11s performed better than YOLO11m for this small dataset, suggesting that smaller models can be more effective for limited data.
2. **Optimizer**: SGD provided slightly better results than AdamW for the segmentation task, with:
   - 2% better box loss
   - 0.4% better segmentation loss
   - 0.1% better mAP50 scores
3. **Training Efficiency**: SGD training was faster and more stable than AdamW.
4. **Best Configuration**: YOLO11s with SGD optimizer and batch size 12 achieved the best overall performance.

### Recommendations:
- For small datasets (<1000 images): Use YOLO11s
- For segmentation tasks: Prefer SGD over AdamW
- Use batch size 12 for optimal performance
- Apply enhanced augmentation techniques for better generalization
