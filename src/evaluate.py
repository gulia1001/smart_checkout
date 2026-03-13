import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import random
from torch.utils.data import Subset
# --- 1. НАСТРОЙКИ ---
# Получаем пути относительно папки src/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DATA_DIR = os.path.join(BASE_DIR, 'dataset', 'raw') 
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Выберите модель, которую хотите протестировать
MODEL_NAME = "MobileNetV2" # Можно поменять на ResNet50, AlexNet и т.д.
MODEL_PATH = os.path.join(MODELS_DIR, f"{MODEL_NAME}_weights1.pth")

BATCH_SIZE = 32

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Используем устройство: {device}")

# --- 2. ИНИЦИАЛИЗАЦИЯ МОДЕЛИ ---
def load_model(model_name, num_classes):
    print(f"🔄 Загрузка архитектуры {model_name}...")
    if model_name == "AlexNet":
        model = models.alexnet(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == "VGG16":
        model = models.vgg16(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == "ResNet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "GoogLeNet":
        model = models.googlenet(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.aux_logits = False
    elif model_name == "MobileNetV2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError("Неизвестная модель!")
    return model

def main():
    if not os.path.exists(TEST_DATA_DIR):
        print(f"❌ Ошибка: Папка с данными не найдена: {TEST_DATA_DIR}")
        return
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Ошибка: Файл весов не найден: {MODEL_PATH}")
        print("Убедитесь, что вы скачали веса из Colab и положили их в папку models/")
        return

    # --- 3. ЗАГРУЗКА ДАННЫХ ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    # 1. Загружаем весь датасет (пока без DataLoader)
    full_dataset = datasets.ImageFolder(TEST_DATA_DIR, transform=transform)

    # 2. Группируем индексы всех картинок по их классам
    class_indices = {i: [] for i in range(len(full_dataset.classes))}
    for idx, target in enumerate(full_dataset.targets):
        class_indices[target].append(idx)

    # 3. Выбираем по 30 случайных индексов из каждой категории
    selected_indices = []
    for indices in class_indices.values():
        random.shuffle(indices) # Перемешиваем, чтобы фото были случайными
        selected_indices.extend(indices[:30]) # Берем ровно 30 (или меньше, если в папке их не хватает)

    # 4. Создаем "обрезанный" датасет и передаем его в DataLoader
    test_dataset = Subset(full_dataset, selected_indices)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"📁 Подготовлено для тестирования: {len(test_dataset)} изображений (по 30 на класс).")
    
    CLASSES = test_dataset.dataset.classes
    num_classes = len(CLASSES)
    print(f"📁 Загружено {len(test_dataset)} изображений для тестирования.")

    # Загружаем веса
    model = load_model(MODEL_NAME, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # --- 4. ОЦЕНКА (INFERENCE) ---
    all_preds = []
    all_labels = []

    print("⏳ Прогоняем изображения через модель... Пожалуйста, подождите.")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- 5. ВЫВОД РЕЗУЛЬТАТОВ ---
    print("\n" + "="*50)
    print(f"📊 CLASSIFICATION REPORT: {MODEL_NAME}")
    print("="*50)
    report = classification_report(all_labels, all_preds, target_names=CLASSES)
    print(report)

    # --- 6. ПОСТРОЕНИЕ CONFUSION MATRIX ---
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(14, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASSES, yticklabels=CLASSES, 
                cbar=False, linewidths=1, linecolor='black')

    plt.title(f'Confusion Matrix: {MODEL_NAME}', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=14, labelpad=10)
    plt.ylabel('True Label', fontsize=14, labelpad=10)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    plt.tight_layout()

    # Сохраняем график в корень проекта
    plot_path = os.path.join(BASE_DIR, f'confusion_matrix_{MODEL_NAME}.png')
    plt.savefig(plot_path, dpi=300)
    print(f"✅ Матрица ошибок сохранена как '{plot_path}'!")

if __name__ == "__main__":
    main()