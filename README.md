# 🛒 Smart Checkout AI: Computer Vision Cashier

This project is an AI-powered automated checkout system. It uses Computer Vision (Sliding Window + CNNs) to detect grocery items on a shelf or table and automatically calculate the total bill.

## 🚀 Features
- **Object Detection:** Custom sliding window implementation with Non-Maximum Suppression (NMS).
- **Multiple Models:** Trained and compared 5 architectures (ResNet50, MobileNetV2, AlexNet, VGG16, GoogLeNet).
- **Web Interface:** Interactive UI built with Streamlit.
- **Smart Data Preparation:** Center-crop preprocessing for better bounding box accuracy.

## 🛠️ Installation & Setup (Local)
1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/smart_checkout.git](https://github.com/YOUR_USERNAME/smart_checkout.git)
   cd smart_checkout