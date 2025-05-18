import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

imgsz = [640, 512]

def predict_and_visualize(model_path: str, image_path: str, output_path: str = None, imgsz: tuple[int, int], conf_threshold: float = 0.25):
    """Run inference and visualize results."""
    # Load the model
    model = YOLO(model_path)
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Run inference
    results = model.predict(image, imgsz=imgsz, conf=conf_threshold)[0]
    
    # Create a copy of the image for visualization
    vis_image = image.copy()
    
    # Draw predictions
    for result in results:
        # Get the mask
        if result.masks is not None:
            mask = result.masks.data[0].cpu().numpy()
            # Resize mask to image size
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
            # Create colored mask
            colored_mask = np.zeros_like(image)
            colored_mask[mask > 0.5] = [0, 255, 0]  # Green mask
            # Blend mask with image
            vis_image = cv2.addWeighted(vis_image, 1, colored_mask, 0.5, 0)
        
        # Draw bounding box
        box = result.boxes.xyxy[0].cpu().numpy()
        cv2.rectangle(vis_image, 
                     (int(box[0]), int(box[1])), 
                     (int(box[2]), int(box[3])), 
                     (0, 0, 255), 2)
        
        # Add confidence score
        conf = result.boxes.conf[0].cpu().numpy()
        cv2.putText(vis_image, 
                   f"{conf:.2f}", 
                   (int(box[0]), int(box[1] - 10)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, 
                   (0, 0, 255), 
                   2)
    
    # Save or display the result
    if output_path:
        cv2.imwrite(output_path, vis_image)
        print(f"Saved visualization to {output_path}")
    else:
        cv2.imshow("Prediction", vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Run inference with trained YOLO model')
    parser.add_argument('--model', type=str, required=True,
                      help='Path to the trained model (e.g., runs/train/exp/weights/best.pt)')
    parser.add_argument('--image', type=str, required=True,
                      help='Path to the input image')
    parser.add_argument('--output', type=str, default=None,
                      help='Path to save the visualization (optional)')
    parser.add_argument('--conf', type=float, default=0.25,
                      help='Confidence threshold (default: 0.25)')
    
    args = parser.parse_args()
    
    try:
        predict_and_visualize(args.model, args.image, args.output, imgsz, args.conf)
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main()) 
