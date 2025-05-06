import os
import xml.etree.ElementTree as ET
import shutil
from pathlib import Path
import numpy as np
from tqdm import tqdm
import random
import argparse
import re
from PIL import Image
import concurrent.futures
from typing import List, Tuple, Optional, Dict
import time
import psutil
import logging
from datetime import datetime, timedelta

# Common PAGE-XML namespaces
PAGE_NS = {
    '2019-07-15': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15',
    '2018-07-15': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2018-07-15',
    '2017-07-15': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2017-07-15',
    '2016-07-15': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2016-07-15',
    '2013-07-15': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15',
    '2010-01-12': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2010-01-12',
    '2009-03-16': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2009-03-16'
}

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

def get_page_ns(xml_file):
    """Extract PAGE namespace from XML file."""
    with open(xml_file, 'r', encoding='utf-8') as f:
        content = f.read()
        # Try to find namespace in XML declaration
        ns_match = re.search(r'xmlns="([^"]+)"', content)
        if ns_match:
            return ns_match.group(1)
        # Try to find namespace in schemaLocation
        schema_match = re.search(r'schemaLocation="[^"]+ ([^"]+)"', content)
        if schema_match:
            return schema_match.group(1)
    return None

def find_image_file(xml_file: Path) -> Optional[Path]:
    """Find corresponding image file for XML file."""
    for ext in SUPPORTED_FORMATS.keys():
        img_file = xml_file.with_suffix(ext)
        if img_file.exists():
            return img_file
    return None

def parse_points(points_str):
    """Parse points string into list of (x,y) tuples."""
    return [(float(x), float(y)) for x, y in [p.split(',') for p in points_str.split()]]

def normalize_points(points, img_width, img_height):
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

def create_polygon_from_baseline(baseline_points, height=50, padding_ratio=0.2):
    """Create a padded polygon from baseline points with given height and padding ratio."""
    if len(baseline_points) < 2:
        return None
    
    # Convert baseline points to numpy array for easier manipulation
    points = np.array(baseline_points)
    
    # Calculate perpendicular vectors for each segment
    segments = points[1:] - points[:-1]
    segment_norms = np.linalg.norm(segments, axis=1)
    
    # Skip if any segment has zero length
    if np.any(segment_norms == 0):
        return None
        
    perp_vectors = np.array([[-seg[1], seg[0]] for seg in segments])
    perp_vectors = perp_vectors / segment_norms[:, np.newaxis]
    
    # Calculate dynamic height based on segment length
    avg_segment_length = np.mean(segment_norms)
    dynamic_height = max(height, avg_segment_length * padding_ratio)
    
    # Create top and bottom points with padding
    top_points = points[:-1] + perp_vectors * dynamic_height/2
    bottom_points = points[:-1] - perp_vectors * dynamic_height/2
    
    # Add horizontal padding at the start
    first_segment = segments[0]
    first_perp = perp_vectors[0]
    first_norm = segment_norms[0]
    if first_norm > 0:  # Check for zero length
        start_padding = min(first_norm * padding_ratio, points[0][0])  # Limit padding to not go negative
        start_point = points[0] - first_segment * (start_padding / first_norm)
    else:
        start_point = points[0]
    
    polygon = [
        start_point + first_perp * dynamic_height/2,
        start_point - first_perp * dynamic_height/2
    ]
    
    # Add middle points
    for i in range(len(points)-1):
        polygon.extend([top_points[i], bottom_points[i]])
    
    # Add horizontal padding at the end
    last_segment = segments[-1]
    last_perp = perp_vectors[-1]
    last_norm = segment_norms[-1]
    if last_norm > 0:  # Check for zero length
        end_padding = min(last_norm * padding_ratio, points[-1][0])  # Limit padding to not go negative
        end_point = points[-1] + last_segment * (end_padding / last_norm)
    else:
        end_point = points[-1]
    
    polygon.extend([
        end_point + last_perp * dynamic_height/2,
        end_point - last_perp * dynamic_height/2
    ])
    
    return polygon

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

def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging configuration."""
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / 'conversion.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class ProcessingStats:
    """Class to track processing statistics."""
    def __init__(self):
        self.start_time = time.time()
        self.processed = 0
        self.failed = 0
        self.retried = 0
        self.total_files = 0
        self.current_file = ""
        self.last_update = time.time()
        self.files_per_second = 0
        self.errors: Dict[str, List[str]] = {}

    def update(self, success: bool, filename: str, error: str = None):
        """Update statistics."""
        self.processed += 1
        if not success:
            self.failed += 1
            if error:
                if error not in self.errors:
                    self.errors[error] = []
                self.errors[error].append(filename)
        
        # Calculate processing speed
        current_time = time.time()
        time_diff = current_time - self.last_update
        if time_diff >= 1.0:  # Update every second
            self.files_per_second = self.processed / (current_time - self.start_time)
            self.last_update = current_time

    def get_progress_info(self) -> str:
        """Get formatted progress information."""
        elapsed = time.time() - self.start_time
        if self.processed > 0:
            eta = (self.total_files - self.processed) / (self.processed / elapsed)
            eta_str = str(timedelta(seconds=int(eta)))
        else:
            eta_str = "Unknown"
        
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        return (
            f"{self.processed}/{self.total_files} "
            f"({self.files_per_second:.1f} files/s) "
            f"ETA: {eta_str} "
            f"Mem: {memory_usage:.0f}MB "
            f"[✓{self.processed - self.failed} ✗{self.failed}]"
        )

def process_file(args: Tuple[Path, Path, str, int, int, float, Optional[int], ProcessingStats, str]) -> bool:
    """Process a single XML file and its corresponding image."""
    xml_file, output_dir, split, height, padding_ratio, train_ratio, target_height, stats, element_type = args
    stats.current_file = xml_file.name
    
    # Find corresponding image file
    img_file = find_image_file(xml_file)
    if not img_file:
        stats.update(False, xml_file.name, "No supported image file found")
        return False
    
    # Get namespace
    ns = get_page_ns(xml_file)
    if not ns:
        stats.update(False, xml_file.name, "Could not determine namespace")
        return False
    
    try:
        # Parse XML
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Get image dimensions
        page = root.find(f'.//{{{ns}}}Page')
        if page is None:
            stats.update(False, xml_file.name, "Could not find Page element")
            return False
            
        img_width = int(page.get('imageWidth'))
        img_height = int(page.get('imageHeight'))
        
        # Preprocess image if target_height is specified
        if target_height:
            output_img_path = output_dir / 'images' / split / img_file.name
            img_width, img_height, width_scale, height_scale = preprocess_image(
                img_file, target_height, output_img_path
            )
        else:
            output_img_path = output_dir / 'images' / split / img_file.name
            shutil.copy2(img_file, output_img_path)
        
        # Process elements
        yolo_labels = []
        if element_type == 'textline':
            elements = root.findall(f'.//{{{ns}}}TextLine')
        else:  # 'zone'
            elements = root.findall(f'.//{{{ns}}}TextRegion')
        for elem in elements:
            if element_type == 'textline':
                coords = elem.find(f'.//{{{ns}}}Coords')
                if coords is not None:
                    points = parse_points(coords.get('points'))
                    if target_height:
                        points = [(x * width_scale, y * height_scale) for x, y in points]
                    polygon = points
                else:
                    polygon = None
            else:  # 'zone'
                coords = elem.find(f'.//{{{ns}}}Coords')
                if coords is not None:
                    points = parse_points(coords.get('points'))
                    if target_height:
                        points = [(x * width_scale, y * height_scale) for x, y in points]
                    polygon = points
                else:
                    polygon = None
            if polygon:
                norm_polygon = normalize_points(polygon, img_width, img_height)
                # Format: class_id x1 y1 x2 y2 ... xn yn with full precision
                yolo_label = '0 ' + ' '.join(f'{x:.16f} {y:.16f}' for x, y in norm_polygon)
                yolo_labels.append(yolo_label)
        
        if yolo_labels:
            label_file = output_dir / 'labels' / split / f"{xml_file.stem}.txt"
            with open(label_file, 'w') as f:
                f.write('\n'.join(yolo_labels))
            stats.update(True, xml_file.name)
            return True
            
    except Exception as e:
        error_msg = f"Error processing {xml_file}: {str(e)}"
        stats.update(False, xml_file.name, error_msg)
        return False
    
    return False

def convert_page_to_yolo(input_dir, output_dir, train_ratio=0.8, height=50, padding_ratio=0.2, target_height=None, element_type='textline'):
    """Convert PAGE-XML files to YOLO11 segmentation format."""
    # Setup logging
    output_dir = Path(output_dir)
    logger = setup_logging(output_dir)
    
    # Create output directory structure
    (output_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (output_dir / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (output_dir / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (output_dir / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
    
    # Get all XML files, excluding METS.xml
    xml_files = [f for f in Path(input_dir).glob('*.xml') if f.name != 'METS.xml']
    if not xml_files:
        logger.error(f"No XML files found in {input_dir}")
        return
    
    random.shuffle(xml_files)
    
    # Initialize statistics
    stats = ProcessingStats()
    stats.total_files = len(xml_files)
    
    # Split into train/val
    split_idx = int(len(xml_files) * train_ratio)
    train_files = xml_files[:split_idx]
    val_files = xml_files[split_idx:]
    
    logger.info(f"Starting conversion of {len(xml_files)} files")
    logger.info(f"Train/Val split: {len(train_files)}/{len(val_files)} files")
    
    # Create a single progress bar for the entire process
    with tqdm(total=len(xml_files), desc="Converting", unit="files", 
             bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}') as pbar:
        # Process train files
        process_args = [
            (xml_file, output_dir, 'train', height, padding_ratio, train_ratio, target_height, stats, element_type)
            for xml_file in train_files
        ]
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_file, args) for args in process_args]
            for future in concurrent.futures.as_completed(futures):
                pbar.update(1)
                pbar.set_postfix_str(stats.get_progress_info())
        
        # Process validation files
        process_args = [
            (xml_file, output_dir, 'val', height, padding_ratio, train_ratio, target_height, stats, element_type)
            for xml_file in val_files
        ]
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_file, args) for args in process_args]
            for future in concurrent.futures.as_completed(futures):
                pbar.update(1)
                pbar.set_postfix_str(stats.get_progress_info())
    
    # Log final statistics
    logger.info("\nProcessing complete!")
    logger.info(f"Successfully processed: {stats.processed - stats.failed} files")
    logger.info(f"Failed: {stats.failed} files")
    logger.info(f"Average speed: {stats.files_per_second:.1f} files/sec")
    
    if stats.errors:
        logger.info("\nError Summary:")
        for error, files in stats.errors.items():
            logger.info(f"\n{error}:")
            for file in files[:5]:  # Show first 5 files for each error type
                logger.info(f"  - {file}")
            if len(files) > 5:
                logger.info(f"  ... and {len(files) - 5} more")
    
    # Create dataset.yaml
    yaml_content = f"""# Dataset configuration for text line segmentation
path: {output_dir}  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')

# Classes
names:
  0: text_line  # class names

# Task
task: segment  # task type: detect, segment, classify, pose

# Image size
imgsz: {target_height if target_height else 640}  # image size for training

# Augmentation
augment:
  hsv_h: 0.015  # HSV-Hue augmentation
  hsv_s: 0.7    # HSV-Saturation augmentation
  hsv_v: 0.4    # HSV-Value augmentation
  degrees: 0.0  # rotation (+/- deg) - disabled to prevent text line distortion
  translate: 0.1  # translation (+/- fraction)
  scale: 0.5    # scale (+/- gain)
  shear: 0.0    # shear (+/- deg) - disabled to prevent text line distortion
  perspective: 0.0  # perspective (+/- fraction) - disabled to prevent text line distortion
  flipud: 0.0    # probability of flip up-down - disabled for text
  fliplr: 0.5    # probability of flip left-right
  mosaic: 0.0    # mosaic probability - disabled to prevent text line splitting
  mixup: 0.0     # mixup probability - disabled to prevent text line splitting
"""
    
    with open(output_dir / 'dataset.yaml', 'w') as f:
        f.write(yaml_content)

def main():
    parser = argparse.ArgumentParser(description='Convert PAGE-XML files to YOLO11 segmentation format')
    parser.add_argument('input_dir', type=str, help='Input directory containing PAGE-XML files and images')
    parser.add_argument('output_dir', type=str, help='Output directory for YOLO11 dataset')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Ratio of training data (default: 0.8)')
    parser.add_argument('--target-height', type=int, help='Target height for resizing images (maintains aspect ratio)')
    parser.add_argument('--element-type', type=str, choices=['textline', 'zone'], default='textline', help='Element type to extract: textline or zone (default: textline)')
    
    args = parser.parse_args()
    
    convert_page_to_yolo(
        args.input_dir,
        args.output_dir,
        train_ratio=args.train_ratio,
        target_height=args.target_height,
        element_type=args.element_type
    )

if __name__ == '__main__':
    main() 