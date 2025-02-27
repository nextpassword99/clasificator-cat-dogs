from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
import cv2
from pathlib import Path
from tensorflow.keras import Sequential, models, layers
import tensorflow as ts
import numpy as np
import tensorflow_datasets as tfds
from PIL import Image


import kagglehub

path_dataset = kagglehub.dataset_download(
    "bhavikjikadara/dog-and-cat-classification-dataset")

print("Path to dataset files:", path_dataset)


data = Path(path_dataset)
print(data)

list(data.glob('**/*.jpg'))[:5]


Image.open(str(list(data.glob('PetImages/Cat/*.*'))[12472]))


Image.open(str(list(data.glob('PetImages/Dog/*.*'))[12472]))

animals = {
    'dog': list(data.glob('PetImages/Dog/*.*')),
    'cat': list(data.glob('PetImages/Cat/*.*')),
}

label = {
    'dog': 1,
    'cat': 0,
}


X = []
Y = []

for name, images in animals.items():
    for image in images:
        try:
            image_path = str(image).strip()
            img = cv2.imread(image_path)

            if image_path is not None:
                img_resize = cv2.resize(img, (200, 200))
                X.append(img_resize)
                Y.append(label[name])
                print(len(X), len(Y))
            else:
                pass
        except Exception as e:
            print(e)


print(len(X), len(Y))

x_np = np.array(X)
y_np = np.array(Y)

X = x_np.astype('float32')
Y = y_np.astype('float32')


print(x_np.shape)


X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y)


model = Sequential([
    layers.Conv2D(32, (3, 3), padding='same',
                  activation='relu', input_shape=(200, 200, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation='relu', kernel_regularizer=l2(0.05)),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu', kernel_regularizer=l2(0.05)),
    layers.Dropout(0.5),

    layers.Dense(1, activation='sigmoid'),
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']

)

model.fit(X_train, y_train, epochs=5)
