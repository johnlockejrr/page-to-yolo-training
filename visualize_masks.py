import os
import cv2
import numpy as np
from pathlib import Path
import argparse
from typing import List, Tuple

def parse_yolo_label(label_path: Path) -> List[List[Tuple[float, float]]]:
    """Parse YOLO format label file and return list of polygons."""
    polygons = []
    with open(label_path, 'r') as f:
        for line in f:
            # Split line into class_id and coordinates
            parts = line.strip().split()
            if len(parts) < 7:  # Need at least class_id and 3 points (6 coordinates)
                continue
            
            # Convert coordinates to float pairs
            coords = [(float(parts[i]), float(parts[i+1])) for i in range(1, len(parts), 2)]
            polygons.append(coords)
    
    return polygons

def draw_polygons(image: np.ndarray, polygons: List[List[Tuple[float, float]]], 
                 color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> np.ndarray:
    """Draw polygons on the image."""
    img_height, img_width = image.shape[:2]
    
    for polygon in polygons:
        # Convert normalized coordinates to pixel coordinates
        points = [(int(x * img_width), int(y * img_height)) for x, y in polygon]
        points = np.array(points, np.int32)
        
        # Draw polygon
        cv2.polylines(image, [points], True, color, thickness)
        
        # Draw points
        for point in points:
            cv2.circle(image, tuple(point), 3, color, -1)
    
    return image

def ensure_output_path(output_path: Path) -> Path:
    """Ensure output path has proper extension and directory exists."""
    # If no extension provided, add .jpg
    if not output_path.suffix:
        output_path = output_path.with_suffix('.jpg')
    
    # Create parent directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    return output_path

def visualize_masks(image_path: Path, label_path: Path, output_path: Path = None, 
                   color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2):
    """Visualize YOLO segmentation masks on an image."""
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Parse label file
    polygons = parse_yolo_label(label_path)
    
    # Draw polygons
    image = draw_polygons(image, polygons, color, thickness)
    
    # Save or show result
    if output_path:
        output_path = ensure_output_path(output_path)
        success = cv2.imwrite(str(output_path), image)
        if success:
            print(f"Saved visualization to: {output_path}")
        else:
            raise ValueError(f"Failed to save image to: {output_path}")
    else:
        # Show image
        cv2.imshow('Segmentation Masks', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Visualize YOLO segmentation masks on images')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('label_path', type=str, help='Path to the YOLO format label file')
    parser.add_argument('--output', '-o', type=str, help='Path to save the visualization (optional)')
    parser.add_argument('--color', type=int, nargs=3, default=[0, 255, 0],
                      help='BGR color for the masks (default: 0 255 0)')
    parser.add_argument('--thickness', type=int, default=2,
                      help='Line thickness for the masks (default: 2)')
    
    args = parser.parse_args()
    
    try:
        visualize_masks(
            Path(args.image_path),
            Path(args.label_path),
            Path(args.output) if args.output else None,
            tuple(args.color),
            args.thickness
        )
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main()) 