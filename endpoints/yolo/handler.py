"""
RunPod Serverless Handler for YOLOv11 Object Detection
"""
import runpod
from ultralytics import YOLO
import base64
import io
import json
from PIL import Image

# Load model once at startup
print("Loading YOLOv11m model...")
model = YOLO('yolo11m.pt')
print("Model loaded successfully!")

def handler(event):
    """
    Handler for YOLOv11 object detection

    Input format:
    {
        "input": {
            "image": "base64_encoded_image",
            "conf": 0.25,  # optional confidence threshold
            "iou": 0.7,    # optional IOU threshold
            "classes": []  # optional: filter by class IDs
        }
    }

    Output format:
    {
        "detections": [
            {
                "bbox": [x1, y1, x2, y2],
                "confidence": 0.95,
                "class": 0,
                "class_name": "person"
            },
            ...
        ],
        "count": 5,
        "inference_time": 0.045
    }
    """
    try:
        import time
        start_time = time.time()

        input_data = event.get('input', {})

        # Decode base64 image
        image_b64 = input_data.get('image')
        if not image_b64:
            return {"error": "No image provided"}

        image_data = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_data))

        # Get parameters
        conf_threshold = input_data.get('conf', 0.25)
        iou_threshold = input_data.get('iou', 0.7)
        filter_classes = input_data.get('classes', None)

        # Run inference
        results = model(
            image,
            conf=conf_threshold,
            iou=iou_threshold,
            classes=filter_classes,
            verbose=False
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
            'image_size': [results.orig_shape[1], results.orig_shape[0]]
        }

    except Exception as e:
        return {"error": str(e), "traceback": __import__('traceback').format_exc()}

# Start the serverless worker
if __name__ == "__main__":
    runpod.serverless.start({'handler': handler})
