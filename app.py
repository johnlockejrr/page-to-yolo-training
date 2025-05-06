import gradio as gr
import os
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

def find_model_checkpoints():
    """Find all best.pt checkpoints in runs/train directory."""
    checkpoints = []
    train_dir = Path("./runs/train")
    
    if not train_dir.exists():
        return []
    
    # Walk through all subdirectories
    for exp_dir in train_dir.iterdir():
        if exp_dir.is_dir():
            weights_dir = exp_dir / "weights"
            if weights_dir.exists():
                best_pt = weights_dir / "best.pt"
                if best_pt.exists():
                    checkpoints.append(str(best_pt))
    
    return checkpoints

def predict_image(model_path, image, show_boxes=False, conf_threshold=0.25):
    """Run prediction and return visualization."""
    if image is None:
        return None
    
    # Load model
    model = YOLO(model_path)
    
    # Convert image to numpy array if it's not already
    if isinstance(image, np.ndarray):
        img = image
    else:
        img = np.array(image)
    
    # Run inference
    results = model.predict(img, conf=conf_threshold)[0]
    
    # Create visualization
    vis_image = img.copy()
    
    # Draw predictions
    for result in results:
        # Draw mask if not showing boxes
        if not show_boxes and result.masks is not None:
            mask = result.masks.data[0].cpu().numpy()
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
            colored_mask = np.zeros_like(img)
            colored_mask[mask > 0.5] = [0, 255, 0]  # Green mask
            vis_image = cv2.addWeighted(vis_image, 1, colored_mask, 0.5, 0)
        
        # Draw bounding box if requested
        if show_boxes:
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
    
    return vis_image

def create_interface():
    """Create and launch the Gradio interface."""
    # Find available models
    checkpoints = find_model_checkpoints()
    
    if not checkpoints:
        print("No model checkpoints found in ./runs/train/")
        return
    
    # Create interface
    with gr.Blocks(title="Text Line Segmentation") as demo:
        gr.Markdown("# Text Line Segmentation Demo")
        
        with gr.Row():
            with gr.Column():
                # Model selection
                model_dropdown = gr.Dropdown(
                    choices=checkpoints,
                    label="Select Model",
                    value=checkpoints[0] if checkpoints else None
                )
                
                # Image upload
                image_input = gr.Image(label="Upload Image", type="numpy")
                
                # Options
                show_boxes = gr.Checkbox(label="Show Bounding Boxes", value=False)
                conf_threshold = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.25,
                    label="Confidence Threshold"
                )
                
                # Predict button
                predict_btn = gr.Button("Predict")
            
            with gr.Column():
                # Output image
                output_image = gr.Image(label="Prediction Result")
        
        # Set up prediction
        predict_btn.click(
            fn=predict_image,
            inputs=[model_dropdown, image_input, show_boxes, conf_threshold],
            outputs=output_image
        )
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    if demo:
        demo.launch() 