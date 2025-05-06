import argparse
from pathlib import Path
import subprocess
import sys
import yaml
import shutil

def update_dataset_yaml(dataset_path: Path) -> Path:
    """Update dataset.yaml with correct paths."""
    yaml_path = dataset_path / 'dataset.yaml'
    if not yaml_path.exists():
        raise FileNotFoundError(f"Dataset configuration not found at {yaml_path}")
    
    # Read the current YAML
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Update paths to be relative to dataset directory
    data['path'] = str(dataset_path.absolute())
    
    # Ensure train and val paths are relative to the dataset directory
    if not data['train'].startswith('images/'):
        data['train'] = f"images/{data['train']}"
    if not data['val'].startswith('images/'):
        data['val'] = f"images/{data['val']}"
    
    # Create a temporary YAML file
    temp_yaml = dataset_path / 'dataset_temp.yaml'
    with open(temp_yaml, 'w') as f:
        yaml.dump(data, f)
    
    # Print the YAML content for debugging
    print("\nDataset configuration:")
    print(f"Dataset path: {data['path']}")
    print(f"Train path: {data['train']}")
    print(f"Val path: {data['val']}")
    print(f"Classes: {data.get('names', {})}\n")
    
    return temp_yaml

def train(args):
    """Train YOLO11 model for segmentation."""
    dataset_path = Path(args.dataset)
    
    # Update dataset.yaml with correct paths
    try:
        yaml_path = update_dataset_yaml(dataset_path)
    except Exception as e:
        print(f"Error updating dataset configuration: {str(e)}")
        return 1
    
    # Verify dataset structure
    train_path = dataset_path / 'images' / 'train'
    val_path = dataset_path / 'images' / 'val'
    if not train_path.exists():
        print(f"Error: Training images directory not found at {train_path}")
        return 1
    if not val_path.exists():
        print(f"Error: Validation images directory not found at {val_path}")
        return 1
    
    # Construct the YOLO command
    cmd = [
        'yolo',
        'segment',  # task
        'train',    # mode
        f'model=yolo11{args.model_size}-seg.pt',
        f'data={yaml_path}',
        f'epochs={args.epochs}',
        f'batch={args.batch_size}',
        f'device={args.device}' if args.device else '',
        f'workers={args.workers}',
        f'project={args.project}',
        f'name={args.name}',
        'exist_ok=True' if args.exist_ok else '',
        'pretrained=True' if args.pretrained else '',
        f'optimizer={args.optimizer}',
        'verbose=True' if args.verbose else '',
        f'seed={args.seed}',
        'deterministic=True' if args.deterministic else '',
        'single_cls=True',  # We're doing text line segmentation
        'rect=True' if args.rect else '',
        'cos_lr=True' if args.cos_lr else '',
        f'close_mosaic={args.close_mosaic}',
        'resume=True' if args.resume else '',
        'amp=True' if args.amp else '',
        f'lr0={args.lr0}',
        f'lrf={args.lrf}',
        f'momentum={args.momentum}',
        f'weight_decay={args.weight_decay}',
        f'warmup_epochs={args.warmup_epochs}',
        f'warmup_momentum={args.warmup_momentum}',
        f'warmup_bias_lr={args.warmup_bias_lr}',
        f'box={args.box}',
        'overlap_mask=True' if args.overlap_mask else '',
        f'mask_ratio={args.mask_ratio}',
        f'dropout={args.dropout}',
        'val=True' if args.val else '',
        'plots=True' if args.plots else ''
    ]
    
    # Remove empty arguments
    cmd = [arg for arg in cmd if arg]
    
    # Run the command
    try:
        subprocess.run(cmd, check=True)
        print("Training completed successfully!")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Error during training: {str(e)}")
        return 1
    finally:
        # Clean up temporary YAML file
        if yaml_path.exists():
            yaml_path.unlink()

def main():
    parser = argparse.ArgumentParser(description='Train YOLO11 model for text line segmentation')
    
    # Required arguments
    parser.add_argument('--dataset', type=str, required=True,
                      help='Path to dataset directory containing dataset.yaml')
    parser.add_argument('--model-size', type=str, choices=['n', 's', 'm', 'l', 'x'],
                      default='s', help='Model size (n=nano, s=small, m=medium, l=large, x=xlarge)')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=16,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of training epochs')
    parser.add_argument('--device', type=str, default='',
                      help='Device to use (e.g., "0" for GPU 0, "cpu" for CPU)')
    parser.add_argument('--workers', type=int, default=8,
                      help='Number of worker threads for data loading')
    
    # Project settings
    parser.add_argument('--project', type=str, default='runs/train',
                      help='Project directory')
    parser.add_argument('--name', type=str, default='exp',
                      help='Experiment name')
    parser.add_argument('--exist-ok', action='store_true',
                      help='Overwrite existing experiment')
    
    # Model settings
    parser.add_argument('--pretrained', action='store_true',
                      help='Use pretrained weights')
    parser.add_argument('--optimizer', type=str, default='auto',
                      choices=['SGD', 'Adam', 'AdamW', 'RMSProp', 'auto'],
                      help='Optimizer to use')
    
    # Training settings
    parser.add_argument('--verbose', action='store_true',
                      help='Print verbose output')
    parser.add_argument('--seed', type=int, default=0,
                      help='Random seed for reproducibility')
    parser.add_argument('--deterministic', action='store_true',
                      help='Enable deterministic training')
    parser.add_argument('--rect', action='store_true',
                      help='Rectangular training')
    parser.add_argument('--cos-lr', action='store_true',
                      help='Use cosine learning rate scheduler')
    parser.add_argument('--close-mosaic', type=int, default=10,
                      help='Disable mosaic augmentation for final epochs')
    parser.add_argument('--resume', action='store_true',
                      help='Resume training from last checkpoint')
    parser.add_argument('--amp', action='store_true',
                      help='Use automatic mixed precision')
    
    # Learning rate settings
    parser.add_argument('--lr0', type=float, default=0.01,
                      help='Initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.01,
                      help='Final learning rate (lr0 * lrf)')
    parser.add_argument('--momentum', type=float, default=0.937,
                      help='SGD momentum/Adam beta1')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                      help='Optimizer weight decay')
    parser.add_argument('--warmup-epochs', type=float, default=3.0,
                      help='Warmup epochs')
    parser.add_argument('--warmup-momentum', type=float, default=0.8,
                      help='Warmup momentum')
    parser.add_argument('--warmup-bias-lr', type=float, default=0.1,
                      help='Warmup bias learning rate')
    
    # Segmentation settings
    parser.add_argument('--box', type=float, default=7.5,
                      help='Box loss gain')
    parser.add_argument('--overlap-mask', action='store_true',
                      help='Masks should overlap during training')
    parser.add_argument('--mask-ratio', type=int, default=4,
                      help='Mask downsample ratio')
    parser.add_argument('--dropout', type=float, default=0.0,
                      help='Use dropout regularization')
    
    # Validation settings
    parser.add_argument('--val', action='store_true',
                      help='Validate training results')
    parser.add_argument('--plots', action='store_true',
                      help='Plot training results')
    
    args = parser.parse_args()
    
    return train(args)

if __name__ == '__main__':
    sys.exit(main()) 