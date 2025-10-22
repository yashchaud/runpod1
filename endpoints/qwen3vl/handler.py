"""
RunPod Serverless Handler for Qwen3-VL-8B-Instruct (October 2025 Release)
Optimized for Google Meet recordings, app demos, UI understanding
Features: 32-language OCR, Visual Agent, Spatial Perception, 256K context
"""
import runpod
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import base64
import io
from PIL import Image

# Load Qwen3-VL-8B-Instruct model
print("Loading Qwen3-VL-8B-Instruct (October 2025)...")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", trust_remote_code=True)

print(f"Model loaded! Device: {model.device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Pre-defined prompts for different analysis modes
PROMPTS = {
    "screen_share": """You are analyzing a screen recording or screen share from a video call (like Google Meet, Zoom, etc.).

Provide a comprehensive analysis including:
1. **Scene Description**: What type of content is being shared (app demo, presentation, code, document, etc.)
2. **Layout**: Describe the screen layout - where are different elements positioned
3. **Text Extraction**: List ALL visible text including:
   - App/window titles
   - Button labels
   - Menu items
   - Text fields and their labels
   - Chat messages
   - Status indicators
   - Any other readable text
4. **UI Elements**: Identify interactive elements like buttons, text fields, dropdowns, etc. with approximate positions
5. **Activity**: What is the user doing or demonstrating
6. **Context**: What's the purpose or goal of this screen share

Be extremely thorough and specific.""",

    "ui_detection": """Analyze this UI and extract detailed information about all interactive elements:

1. List each UI element with:
   - Type (button, text field, dropdown, checkbox, etc.)
   - Label/text
   - Approximate position (top-left, center, bottom-right, etc.)
   - State (enabled/disabled, checked/unchecked, active/inactive)

2. Describe the layout and spatial relationships

3. Identify the application or platform if recognizable

Format as structured data.""",

    "ocr_extract": """Extract ALL visible text from this image. Include:
- Headings and titles
- Body text
- Button labels
- Menu items
- Form field labels
- Status messages
- Any other visible text

Organize by screen regions (top, middle, bottom) and preserve text hierarchy.""",

    "meeting_analysis": """Analyze this video call/meeting screenshot:

1. Number of participants visible
2. Who is presenting/screen sharing (if applicable)
3. What content is being shared
4. Meeting platform (Google Meet, Zoom, Teams, etc.)
5. Participant names if visible
6. Any chat messages or reactions visible
7. Overall meeting context and activity""",

    "app_demo": """This appears to be an application demonstration. Analyze:

1. What application is being demonstrated
2. What feature or workflow is being shown
3. Step-by-step description of what's visible
4. UI elements and their purpose
5. Text/labels visible
6. User journey or flow being demonstrated
7. Any important details for understanding the app's functionality"""
}

def handler(event):
    """
    Handler for Qwen3-VL-8B vision-language understanding

    Optimized for: Screen recordings, Google Meet calls, app demos, UI analysis

    Input format:
    {
        "input": {
            "image": "base64_encoded_image",  # or "images": [...] for batch
            "mode": "screen_share",  # or "ui_detection", "ocr_extract", "meeting_analysis", "app_demo"
            "custom_prompt": "optional custom prompt",
            "max_tokens": 2048
        }
    }

    Output format:
    {
        "response": "Detailed analysis...",
        "processing_time": 2.5
    }
    """
    try:
        import time
        start_time = time.time()

        input_data = event.get('input', {})

        # Check if batch or single
        is_batch = 'images' in input_data

        # Get mode and prompt
        mode = input_data.get('mode', 'screen_share')
        custom_prompt = input_data.get('custom_prompt')
        max_tokens = input_data.get('max_tokens', 2048)

        # Select system prompt
        system_prompt = custom_prompt if custom_prompt else PROMPTS.get(mode, PROMPTS['screen_share'])

        if is_batch:
            # BATCH PROCESSING
            images_b64 = input_data.get('images', [])
            if not images_b64:
                return {"error": "No images provided"}

            results = []

            for idx, img_b64 in enumerate(images_b64):
                # Decode image
                image_data = base64.b64decode(img_b64)
                image = Image.open(io.BytesIO(image_data)).convert('RGB')

                # Prepare messages
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": system_prompt}
                        ]
                    }
                ]

                # Process
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)

                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt"
                ).to(model.device)

                # Generate with Qwen3-VL optimizations
                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=False,
                        temperature=0.7
                    )

                # Decode
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                response = processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]

                results.append({
                    'frame_index': idx,
                    'response': response,
                    'mode': mode
                })

            processing_time = time.time() - start_time

            return {
                'responses': results,
                'batch_size': len(images_b64),
                'processing_time': processing_time,
                'model': 'Qwen3-VL-8B-Instruct'
            }

        else:
            # SINGLE IMAGE
            image_b64 = input_data.get('image')
            if not image_b64:
                return {"error": "No image provided"}

            # Decode image
            image_data = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_data)).convert('RGB')

            # Prepare messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": system_prompt}
                    ]
                }
            ]

            # Process
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(model.device)

            # Generate
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    temperature=0.7
                )

            # Decode
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

            processing_time = time.time() - start_time

            return {
                'response': response,
                'processing_time': processing_time,
                'image_size': image.size,
                'mode': mode,
                'model': 'Qwen3-VL-8B-Instruct'
            }

    except Exception as e:
        return {"error": str(e), "traceback": __import__('traceback').format_exc()}

if __name__ == "__main__":
    runpod.serverless.start({'handler': handler})
