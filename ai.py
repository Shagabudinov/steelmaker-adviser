import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Задайте путь к папке с изображениями игр
dataset_path = "D:\Dataset"

# Задайте размеры изображений
image_width, image_height = 64, 64

# Загрузка и обработка изображений
def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = load_img(os.path.join(folder, filename), target_size=(image_width, image_height))
        img_array = img_to_array(img)
        images.append(img_array)
        labels.append(os.path.basename(folder))  # Используем имя папки как метку класса
    return np.array(images), np.array(labels)

# Загрузка и подготовка данных
images = []
labels = []
for game_folder in os.listdir(dataset_path):
    game_folder_path = os.path.join(dataset_path, game_folder)
    game_images, game_labels = load_images_from_folder(game_folder_path)
    images.extend(game_images)
    labels.extend(game_labels)

images = np.array(images)
labels = np.array(labels)

# Нормализация значений пикселей от 0 до 1
images = images.astype('float32') / 255.0

# Создание словаря соответствия названия игры и ее числовой метки
game_to_label = {game: i for i, game in enumerate(np.unique(labels))}
num_classes = len(game_to_label)

# Преобразование меток в числовой формат
numeric_labels = np.array([game_to_label[game] for game in labels])

# Разделение данных на обучающую и тестовую выборки (80% - обучающая, 20% - тестовая)
split_index = int(0.8 * len(images))
train_images, test_images = images[:split_index], images[split_index:]
train_labels, test_labels = numeric_labels[:split_index], numeric_labels[split_index:]

# Создание модели нейронной сети
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image_width, image_height, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Компиляция модели
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Обучение модели
model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Оценка модели на тестовых данных
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print("Точность на тестовых данных:", test_acc)
