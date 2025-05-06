# Text Line Segmentation with YOLO11

This repository contains scripts for converting PAGE-XML annotations to YOLO format and training a YOLO11 model for text line segmentation.

## Scripts

### 1. `convert_page_to_yolo.py`
Converts PAGE-XML annotations to YOLO format for segmentation training.

```bash
python convert_page_to_yolo.py input_dir output_dir --target-height 640 --element-type textline
```

Arguments:
- `input_dir`: Directory containing PAGE-XML files
- `output_dir`: Directory to save YOLO format annotations
- `--target-height`: Target image height (default: 640)
- `--element-type`: Type of element to extract ('textline' or 'zone')

### 2. `visualize_masks.py`
Visualizes YOLO segmentation masks on images.

```bash
python visualize_masks.py image_path label_path [--output output_path] [--color B G R] [--thickness N]
```

Arguments:
- `image_path`: Path to the input image
- `label_path`: Path to the YOLO format label file
- `--output`: Path to save the visualization (optional)
- `--color`: BGR color values for the mask (default: 0 255 0)
- `--thickness`: Line thickness for the mask (default: 2)

### 3. `train.py`
Trains a YOLO11 model for text line segmentation.

```bash
python train.py --dataset dataset_path --model-size m --batch-size 8
```

Required arguments:
- `--dataset`: Path to dataset directory containing dataset.yaml
- `--model-size`: Model size (n=nano, s=small, m=medium, l=large, x=xlarge)
- `--batch-size`: Batch size for training

Optional arguments:
- `--epochs`: Number of training epochs (default: 100)
- `--device`: Device to use (e.g., "0" for GPU 0, "cpu" for CPU)
- `--workers`: Number of worker threads (default: 8)
- `--project`: Project directory (default: runs/train)
- `--name`: Experiment name (default: exp)
- `--pretrained`: Use pretrained weights
- `--optimizer`: Optimizer to use (default: auto)
- `--amp`: Use automatic mixed precision
- `--val`: Validate training results
- `--plots`: Plot training results

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

## Training Progress

The training script provides detailed metrics:
- Box and segmentation losses
- Precision and recall for both detection and segmentation
- mAP50 and mAP50-95 scores
- GPU memory usage and training speed

## Visualization

The visualization script helps verify:
- Text line detection accuracy
- Segmentation mask quality
- Overall model performance

## Notes

- The conversion script preserves original polygon shapes without padding
- Training uses single-class segmentation for text lines
- The model supports various sizes (nano to xlarge) for different performance requirements 