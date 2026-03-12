import os
from PIL import Image
from tqdm import tqdm  # Обычный tqdm для локального терминала

# --- НАСТРОЙКИ ПУТЕЙ (относительно папки src) ---
# Предполагается, что скрипт лежит в папке src/, а данные в dataset/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_DIR = os.path.join(BASE_DIR, 'dataset', 'raw')
TARGET_DIR = os.path.join(BASE_DIR, 'dataset', 'processed')

TARGET_SIZE = (224, 224)

# Классы, которые нужно обрезать сильнее (маленькие объекты)
SMALL_OBJECTS = ['candybars', 'chocolate']

def main():
    os.makedirs(TARGET_DIR, exist_ok=True)

    try:
        classes = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]
        print(f"📁 Найдено классов: {len(classes)}. Начинаем умную обрезку и сжатие...")
    except FileNotFoundError:
        print(f"❌ Ошибка: Папка {SOURCE_DIR} не найдена.")
        print("Убедитесь, что вы положили оригинальные фото в папку dataset/raw/")
        return

    total_images = 0

    for cls_name in classes:
        source_class_dir = os.path.join(SOURCE_DIR, cls_name)
        target_class_dir = os.path.join(TARGET_DIR, cls_name)
        os.makedirs(target_class_dir, exist_ok=True)

        images = [f for f in os.listdir(source_class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Определяем, насколько сильно обрезать края (оставляем X% в центре)
        crop_factor = 0.45 if cls_name in SMALL_OBJECTS else 0.65

        # tqdm без leave=False, чтобы логи в терминале выглядели красиво
        for img_name in tqdm(images, desc=f"Обработка: {cls_name}"):
            source_path = os.path.join(source_class_dir, img_name)
            target_path = os.path.join(target_class_dir, img_name)

            try:
                with Image.open(source_path) as img:
                    img = img.convert('RGB')

                    # --- УМНАЯ ОБРЕЗКА ЦЕНТРА ---
                    width, height = img.size

                    # Вычисляем новые размеры
                    new_width = width * crop_factor
                    new_height = height * crop_factor

                    # Вычисляем координаты для вырезания центрального прямоугольника
                    left = (width - new_width) / 2
                    top = (height - new_height) / 2
                    right = (width + new_width) / 2
                    bottom = (height + new_height) / 2

                    # Вырезаем объект
                    img = img.crop((left, top, right, bottom))

                    # --- СЖАТИЕ ---
                    img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
                    img.save(target_path, format='JPEG', quality=85)
                    total_images += 1
            except Exception as e:
                print(f"⚠️ Ошибка с файлом {img_name}: {e}")

    print(f"\n✅ Готово! Успешно обработано и сохранено в {TARGET_DIR} изображений: {total_images}.")

if __name__ == "__main__":
    main()