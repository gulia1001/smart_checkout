import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import os
import copy

# --- 1. НАСТРОЙКИ ПУТЕЙ (ЛОКАЛЬНЫЕ) ---
# Получаем абсолютный путь к главной папке проекта (на уровень выше папки src)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Откуда берем готовые (обрезанные) картинки
DATA_DIR = os.path.join(BASE_DIR, 'dataset', 'processed')

# Куда сохраняем обученные веса моделей (.pth)
SAVE_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(SAVE_DIR, exist_ok=True)

EPOCHS = 15
BATCH_SIZE = 32

# Проверяем, доступна ли видеокарта (GPU) или работаем на процессоре (CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Используем устройство: {device}")

# --- 2. ПОДГОТОВКА ДАННЫХ (DataLoaders) ---
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

try:
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=data_transforms)
except FileNotFoundError:
    print(f"❌ Ошибка: Папка с данными не найдена по пути {DATA_DIR}.")
    print("Убедитесь, что вы сначала запустили скрипт prepare_data.py!")
    exit()

class_names = full_dataset.classes
num_classes = len(class_names)
print(f"📁 Найдено {num_classes} классов: {class_names}")

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

dataloaders = {
    'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
    'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
}

# --- 3. ФУНКЦИЯ ДЛЯ НАСТРОЙКИ МОДЕЛЕЙ (Transfer Learning) ---
def set_parameter_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = False

def initialize_model(model_name, num_classes):
    if model_name == "AlexNet":
        model_ft = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        set_parameter_requires_grad(model_ft)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)

    elif model_name == "VGG16":
        model_ft = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        set_parameter_requires_grad(model_ft)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)

    elif model_name == "ResNet50":
        model_ft = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        set_parameter_requires_grad(model_ft)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "GoogLeNet":
        model_ft = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
        set_parameter_requires_grad(model_ft)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        model_ft.aux_logits = False

    elif model_name == "MobileNetV2":
        model_ft = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        set_parameter_requires_grad(model_ft)
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes)

    return model_ft.to(device)

# --- 4. ЦИКЛ ОБУЧЕНИЯ ---
def train_model(model, dataloaders, criterion, optimizer, num_epochs=15):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs} | ', end='')

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]

                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | ', end='')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()

    print(f'🏆 Лучшая валидационная точность: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model

# --- 5. ЗАПУСК ОБУЧЕНИЯ ДЛЯ ВСЕХ МОДЕЛЕЙ ---
def main():
    models_to_train = ["AlexNet", "VGG16", "GoogLeNet", "ResNet50", "MobileNetV2"]
    criterion = nn.CrossEntropyLoss()

    for model_name in models_to_train:
        print(f"\n{'='*40}")
        print(f"🚀 ОБУЧЕНИЕ: {model_name}")
        print(f"{'='*40}")

        model = initialize_model(model_name, num_classes)
        params_to_update = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(params_to_update, lr=0.001)

        trained_model = train_model(model, dataloaders, criterion, optimizer, num_epochs=EPOCHS)

        save_path = os.path.join(SAVE_DIR, f"{model_name}_weights1.pth")
        torch.save(trained_model.state_dict(), save_path)
        print(f"✅ Модель {model_name} сохранена в {save_path}")

        # Очистка памяти
        del model
        del trained_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n🎉 ВСЕ 5 МОДЕЛЕЙ УСПЕШНО ОБУЧЕНЫ НА PYTORCH И СОХРАНЕНЫ ЛОКАЛЬНО!")

if __name__ == "__main__":
    main()