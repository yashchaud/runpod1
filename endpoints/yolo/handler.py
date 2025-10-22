"""
RunPod Serverless Handler for YOLOv11 Object Detection - BATCH ENABLED
Optimized for 24GB VRAM with batch processing support
"""
import runpod
from ultralytics import YOLO
import base64
import io
import json
from PIL import Image
import torch

# Load model once at startup with optimizations
print("Loading YOLOv11m model with batch optimization...")
model = YOLO('yolo11m.pt')
# Enable GPU optimizations
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print("Model loaded successfully!")

def handler(event):
    """
    Handler for YOLOv11 object detection with BATCH support

    Input format (SINGLE):
    {
        "input": {
            "image": "base64_encoded_image",
            "conf": 0.15,  # Lower threshold for more detail
            "iou": 0.5,    # Lower IOU for more detections
            "imgsz": 1280, # Higher resolution for better detail
            "classes": []  # optional: filter by class IDs
        }
    }

    Input format (BATCH):
    {
        "input": {
            "images": ["base64_1", "base64_2", ...],  # List of images
            "conf": 0.15,
            "iou": 0.5,
            "imgsz": 1280
        }
    }

    Output format:
    {
        "detections": [...] or [[...], [...], ...]  # Single or batch results
        "count": 5 or [5, 3, 7, ...]
        "inference_time": 0.045
        "batch_size": 1 or N
    }
    """
    try:
        import time
        start_time = time.time()

        input_data = event.get('input', {})

        # Check if batch or single
        is_batch = 'images' in input_data

        # Get parameters
        conf_threshold = input_data.get('conf', 0.15)  # Lower for more detail
        iou_threshold = input_data.get('iou', 0.5)     # Lower for more detections
        imgsz = input_data.get('imgsz', 1280)           # Higher res
        filter_classes = input_data.get('classes', None)

        if is_batch:
            # BATCH PROCESSING
            images_b64 = input_data.get('images', [])
            if not images_b64:
                return {"error": "No images provided"}

            # Decode all images
            images = []
            for img_b64 in images_b64:
                image_data = base64.b64decode(img_b64)
                image = Image.open(io.BytesIO(image_data))
                images.append(image)

            # Run batch inference (YOLO automatically batches)
            results = model(
                images,
                conf=conf_threshold,
                iou=iou_threshold,
                imgsz=imgsz,
                classes=filter_classes,
                verbose=False,
                # Optimizations for batch processing
                half=True,  # Use FP16 for faster processing
                device=0    # Use GPU 0
            )

            # Format batch detections
            all_detections = []
            all_counts = []

            for result in results:
                detections = []
                for box in result.boxes:
                    detections.append({
                        'bbox': box.xyxy[0].cpu().numpy().tolist(),
                        'confidence': float(box.conf[0].cpu().numpy()),
                        'class': int(box.cls[0].cpu().numpy()),
                        'class_name': model.names[int(box.cls[0])]
                    })
                all_detections.append(detections)
                all_counts.append(len(detections))

            inference_time = time.time() - start_time

            return {
                'detections': all_detections,
                'count': all_counts,
                'inference_time': inference_time,
                'batch_size': len(images),
                'throughput': len(images) / inference_time
            }

        else:
            # SINGLE IMAGE PROCESSING (backward compatible)
            image_b64 = input_data.get('image')
            if not image_b64:
                return {"error": "No image provided"}

            image_data = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_data))

            # Run inference
            results = model(
                image,
                conf=conf_threshold,
                iou=iou_threshold,
                imgsz=imgsz,
                classes=filter_classes,
                verbose=False,
                half=True,
                device=0
            )[0]

            # Format detections
            detections = []
            for box in results.boxes:
                detections.append({
                    'bbox': box.xyxy[0].cpu().numpy().tolist(),
                    'confidence': float(box.conf[0].cpu().numpy()),
                    'class': int(box.cls[0].cpu().numpy()),
                    'class_name': model.names[int(box.cls[0])]
                })

            inference_time = time.time() - start_time

            return {
                'detections': detections,
                'count': len(detections),
                'inference_time': inference_time,
                'image_size': [results.orig_shape[1], results.orig_shape[0]],
                'batch_size': 1
            }

    except Exception as e:
        return {"error": str(e), "traceback": __import__('traceback').format_exc()}

# Start the serverless worker
if __name__ == "__main__":
    runpod.serverless.start({'handler': handler})
