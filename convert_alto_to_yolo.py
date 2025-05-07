import xml.etree.ElementTree as ET
import os
import argparse
from pathlib import Path
import shutil
import yaml
from PIL import Image
import numpy as np
from tqdm import tqdm
import logging
import sys
import re
from typing import List, Tuple, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Supported image formats
SUPPORTED_FORMATS = {
    '.jpg': 'JPEG',
    '.jpeg': 'JPEG',
    '.png': 'PNG',
    '.tif': 'TIFF',
    '.tiff': 'TIFF',
    '.bmp': 'BMP',
    '.webp': 'WEBP'
}

def parse_points(points_str: str) -> List[Tuple[float, float]]:
    """Parse points string into list of (x,y) tuples.
    
    Args:
        points_str: Space-separated string of x y coordinates (e.g., "x1 y1 x2 y2 x3 y3")
                   Each pair of numbers represents an (x,y) coordinate
    
    Returns:
        List of (x,y) coordinate tuples
    """
    # Split into individual numbers
    numbers = points_str.split()
    # Create pairs of (x,y) coordinates
    return [(float(numbers[i]), float(numbers[i + 1])) for i in range(0, len(numbers), 2)]

def normalize_points(points: List[Tuple[float, float]], img_width: int, img_height: int) -> List[Tuple[float, float]]:
    """Normalize points to [0,1] range and ensure they stay within bounds."""
    normalized = []
    for x, y in points:
        # Normalize coordinates
        norm_x = x / img_width
        norm_y = y / img_height
        
        # Ensure coordinates stay within [0,1]
        norm_x = max(0.0, min(1.0, norm_x))
        norm_y = max(0.0, min(1.0, norm_y))
        
        normalized.append((norm_x, norm_y))
    return normalized

class AltoParser:
    def __init__(self, xml_file_path):
        self.tree = ET.parse(xml_file_path)
        self.root = self.tree.getroot()
        # Extracting the namespace from the XML file
        namespace_uri = self.root.tag.split('}')[0].strip('{')
        self.ns = {'alto': namespace_uri}

    def parse_text_lines(self):
        for text_line in self.root.findall('.//alto:TextLine', self.ns):
            # Get the Shape element with Polygon points
            shape = text_line.find('.//alto:Shape/alto:Polygon', self.ns)
            if shape is not None:
                points_str = shape.get('POINTS')
                if points_str:
                    # Parse points string into list of (x, y) coordinates
                    points = parse_points(points_str)
                    
                    # Get bounding box for YOLO format
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]
                    hpos = min(x_coords)
                    vpos = min(y_coords)
                    width = max(x_coords) - hpos
                    height = max(y_coords) - vpos
                    
                    # Get text content
                    line_text = ' '.join([string.get('CONTENT') for string in text_line.findall('alto:String', self.ns)])
                    
                    yield hpos, vpos, width, height, points, line_text

def convert_coordinates_to_yolo(x, y, w, h, img_width, img_height):
    """Convert coordinates to YOLO format (normalized)."""
    # Calculate center points and dimensions
    x_center = (x + w/2) / img_width
    y_center = (y + h/2) / img_height
    width = w / img_width
    height = h / img_height
    
    return x_center, y_center, width, height

def preprocess_image(img_path: Path, target_height: int, output_path: Path) -> Tuple[int, int, float, float]:
    """Resize image maintaining aspect ratio and save it."""
    with Image.open(img_path) as img:
        # Convert to RGB if necessary
        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            img = img.convert('RGB')
        
        # Calculate new width maintaining aspect ratio
        aspect_ratio = img.width / img.height
        new_width = int(target_height * aspect_ratio)
        
        # Resize image
        resized_img = img.resize((new_width, target_height), Image.Resampling.LANCZOS)
        
        # Calculate scaling factors
        width_scale = new_width / img.width
        height_scale = target_height / img.height
        
        # Save with appropriate format and quality
        output_format = SUPPORTED_FORMATS[output_path.suffix.lower()]
        save_kwargs = {'format': output_format}
        
        if output_format == 'JPEG':
            save_kwargs['quality'] = 95
        elif output_format == 'PNG':
            save_kwargs['optimize'] = True
        elif output_format == 'TIFF':
            save_kwargs['compression'] = 'tiff_lzw'
        
        resized_img.save(output_path, **save_kwargs)
        
        return new_width, target_height, width_scale, height_scale

def process_image(xml_path, image_path, output_dir, target_height=640):
    """Process a single image and its ALTO-XML file."""
    try:
        # Create output directories
        images_dir = os.path.join(output_dir, 'images')
        labels_dir = os.path.join(output_dir, 'labels')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        # Parse ALTO-XML
        parser = AltoParser(xml_path)
        
        # Preprocess image
        image_path = Path(image_path)
        output_path = Path(images_dir) / image_path.name
        new_width, new_height, width_scale, height_scale = preprocess_image(image_path, target_height, output_path)
        
        # Create label file
        label_filename = os.path.splitext(image_path.name)[0] + '.txt'
        label_output_path = os.path.join(labels_dir, label_filename)
        
        yolo_labels = []
        with open(label_output_path, 'w') as f:
            for hpos, vpos, width, height, polygon_points, _ in parser.parse_text_lines():
                # Scale polygon points
                if target_height:
                    points = [(x * width_scale, y * height_scale) for x, y in polygon_points]
                else:
                    points = polygon_points
                
                # Normalize points
                norm_points = normalize_points(points, new_width, new_height)
                
                # Format: class_id x1 y1 x2 y2 ... xn yn with full precision
                yolo_label = '0 ' + ' '.join(f'{x:.16f} {y:.16f}' for x, y in norm_points)
                yolo_labels.append(yolo_label)
            
            # Write all labels
            f.write('\n'.join(yolo_labels))
        
        return True
    except Exception as e:
        logger.error(f"Error processing {xml_path}: {str(e)}")
        return False

def create_dataset_yaml(output_dir):
    """Create dataset.yaml file."""
    yaml_content = {
        'path': str(Path(output_dir).absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'names': {
            0: 'textline'
        }
    }
    
    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f)

def split_dataset(output_dir, val_split=0.2):
    """Split dataset into train and validation sets."""
    images_dir = os.path.join(output_dir, 'images')
    labels_dir = os.path.join(output_dir, 'labels')
    
    # Create train and val directories
    for split in ['train', 'val']:
        os.makedirs(os.path.join(images_dir, split), exist_ok=True)
        os.makedirs(os.path.join(labels_dir, split), exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(images_dir) if f.endswith(tuple(SUPPORTED_FORMATS.keys()))]
    
    # Shuffle and split
    np.random.shuffle(image_files)
    split_idx = int(len(image_files) * (1 - val_split))
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    # Move files to respective directories
    for files, split in [(train_files, 'train'), (val_files, 'val')]:
        for img_file in tqdm(files, desc=f"Moving {split} files"):
            # Move image
            src_img = os.path.join(images_dir, img_file)
            dst_img = os.path.join(images_dir, split, img_file)
            shutil.move(src_img, dst_img)
            
            # Move corresponding label
            label_file = os.path.splitext(img_file)[0] + '.txt'
            src_label = os.path.join(labels_dir, label_file)
            dst_label = os.path.join(labels_dir, split, label_file)
            shutil.move(src_label, dst_label)

def main():
    parser = argparse.ArgumentParser(description='Convert ALTO-XML files to YOLO11 segmentation format')
    parser.add_argument('input_dir', help='Input directory containing ALTO-XML files and images')
    parser.add_argument('output_dir', help='Output directory for YOLO11 dataset')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                      help='Ratio of training data (default: 0.8)')
    parser.add_argument('--target-height', type=int, default=640,
                      help='Target height for resizing images (maintains aspect ratio)')
    parser.add_argument('--element-type', choices=['textline', 'zone'], default='textline',
                      help='Element type to extract: textline or zone (default: textline)')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get all XML files
    xml_files = [f for f in os.listdir(args.input_dir) if f.endswith('.xml')]
    total_files = len(xml_files)
    
    if total_files == 0:
        logger.error("No XML files found in the input directory!")
        return
    
    logger.info(f"Found {total_files} XML files to process")
    
    # Process each XML file with progress bar
    successful = 0
    failed = 0
    
    with tqdm(total=total_files, desc="Processing files") as pbar:
        for file_name in xml_files:
            xml_path = os.path.join(args.input_dir, file_name)
            base_name = os.path.splitext(file_name)[0]
            
            # Find corresponding image file
            image_extensions = list(SUPPORTED_FORMATS.keys())
            image_path = None
            for ext in image_extensions:
                potential_path = os.path.join(args.input_dir, base_name + ext)
                if os.path.exists(potential_path):
                    image_path = potential_path
                    break
            
            if image_path:
                if process_image(xml_path, image_path, args.output_dir, args.target_height):
                    successful += 1
                else:
                    failed += 1
            else:
                logger.warning(f"No matching image found for {file_name}")
                failed += 1
            
            pbar.update(1)
            pbar.set_postfix({'success': successful, 'failed': failed})
    
    # Split dataset into train and validation sets
    logger.info("Splitting dataset into train and validation sets...")
    split_dataset(args.output_dir, 1 - args.train_ratio)  # Convert train_ratio to val_split
    
    # Create dataset.yaml
    logger.info("Creating dataset.yaml...")
    create_dataset_yaml(args.output_dir)
    
    logger.info(f"Conversion completed! Successfully processed {successful} files, {failed} files failed.")

if __name__ == '__main__':
    main() 