import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
import torchvision.ops as ops
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time
import os

# --- 1. SETTINGS & CONSTANTS ---
CLASSES = ['apple', 'background', 'banana', 'bread', 'candybars', 'cereal', 'chips',
           'chocolate', 'cola', 'juice', 'milk', 'noodles', 'orange', 'pickles', 'water', 'yogurt']

PRICES = {
    'apple': 100, 'banana': 200, 'bread': 200, 'candybars': 350, 'cereal': 2000,
    'chips': 600, 'chocolate': 700, 'cola': 600, 'juice': 500, 'milk': 600,
    'noodles': 700, 'orange': 200, 'pickles': 1500, 'water': 350, 'yogurt': 400
}

WINDOW_SIZE = (224, 224)

# Streamlit Page Config
st.set_page_config(page_title="Smart Checkout AI", page_icon="🛒", layout="wide")

# --- 2. CACHED MODEL LOADING ---
@st.cache_resource
def load_model(model_name, model_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(model_path):
        return None, None, f"Error: Weights file not found at {model_path}"

    try:
        if model_name == "MobileNetV2":
            model = models.mobilenet_v2(weights=None)
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, len(CLASSES))
        elif model_name == "ResNet50":
            model = models.resnet50(weights=None)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, len(CLASSES))
        else:
            return None, None, "Model architecture not supported in this demo yet."

        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model, device, "Success"
    except Exception as e:
        return None, None, str(e)

# --- 3. UI LAYOUT ---
st.title("🛒 Smart Checkout: AI Cashier")
st.markdown("Upload a picture of grocery items, and the AI will detect them, draw bounding boxes, and calculate your total bill!")

# Sidebar for controls
st.sidebar.header("⚙️ Settings")
selected_model = st.sidebar.selectbox("Choose AI Model", ["MobileNetV2", "ResNet50"])

# ⚠️ CORRECTED PATH DIRECTION HERE ⚠️
# This forces the script to look for the "models" folder relative to where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
weights_path = os.path.join(BASE_DIR, "models", f"{selected_model}_weights1.pth")

conf_threshold = st.sidebar.slider("Confidence Threshold", 0.50, 0.99, 0.85, 0.01)
nms_threshold = st.sidebar.slider("NMS Threshold (Overlap)", 0.10, 0.90, 0.30, 0.05)
step_size = st.sidebar.slider("Sliding Window Step Size", 32, 128, 64, 16)

# Load the model
model, device, status = load_model(selected_model, weights_path)
if model is None:
    st.sidebar.error(status)
else:
    st.sidebar.success(f"{selected_model} loaded on {device}!")

# Main area for image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    # Display the uploaded image
    col1, col2 = st.columns(2)

    original_image = Image.open(uploaded_file).convert('RGB')
    with col1:
        st.subheader("Original Image")
        st.image(original_image, use_column_width=True)

    if st.button("🚀 Run Detection", type="primary"):
        with st.spinner("Scanning image with Sliding Window... This might take a few seconds."):
            # Prepare image and transforms
            image_np = np.array(original_image)
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            boxes, scores, labels = [], [], []

            # Sliding Window Logic
            for y in range(0, image_np.shape[0] - WINDOW_SIZE[1] + 1, step_size):
                for x in range(0, image_np.shape[1] - WINDOW_SIZE[0] + 1, step_size):
                    patch = image_np[y:y + WINDOW_SIZE[1], x:x + WINDOW_SIZE[0]]
                    pil_patch = Image.fromarray(patch)
                    input_tensor = transform(pil_patch).unsqueeze(0).to(device)

                    with torch.no_grad():
                        outputs = model(input_tensor)
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                        max_prob, predicted_idx = torch.max(probabilities, 0)
                        prob = max_prob.item()
                        class_idx = predicted_idx.item()
                        class_name = CLASSES[class_idx]

                        if prob > conf_threshold and class_name != 'background':
                            boxes.append([x, y, x + WINDOW_SIZE[0], y + WINDOW_SIZE[1]])
                            scores.append(prob)
                            labels.append(class_idx)

            # NMS and Drawing
            result_image = original_image.copy()
            draw = ImageDraw.Draw(result_image)

            total_items = 0
            total_price = 0.0
            receipt_items = []

            if len(boxes) > 0:
                boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
                scores_tensor = torch.tensor(scores, dtype=torch.float32)
                keep_indices = ops.nms(boxes_tensor, scores_tensor, nms_threshold)

                final_boxes = boxes_tensor[keep_indices].numpy()
                final_scores = scores_tensor[keep_indices].numpy()
                final_labels = [labels[i] for i in keep_indices]

                total_items = len(final_boxes)

                for i in range(total_items):
                    x_min, y_min, x_max, y_max = final_boxes[i]
                    class_name = CLASSES[final_labels[i]]
                    score = final_scores[i]
                    price = PRICES.get(class_name, 0.0)

                    total_price += price
                    receipt_items.append({"Item": class_name.capitalize(), "Confidence": f"{score:.2f}", "Price": f"{price:.2f}tg"})

                    # Draw box
                    draw.rectangle([x_min, y_min, x_max, y_max], outline="lime", width=4)
                    
                    # Fix for drawing the label background (avoid negative coordinates)
                    label_text = f"{class_name} {price} tg"
                    draw.rectangle([x_min, max(0, y_min-20), x_min+120, y_min], fill="red")
                    draw.text((x_min+5, max(0, y_min-15)), label_text, fill="white")

            with col2:
                st.subheader("Detection Result")
                st.image(result_image, use_column_width=True)

            # Print Receipt
            st.divider()
            st.subheader("🧾 Final Receipt")
            st.metric("Total Items", total_items)
            st.metric("Total Price", f"{total_price:.2f} tg")

            if len(receipt_items) > 0:
                st.table(receipt_items)
            else:
                st.warning("No items detected. Try lowering the Confidence Threshold or checking your background.")