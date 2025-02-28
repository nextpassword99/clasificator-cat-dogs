from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
import cv2
from pathlib import Path
from tensorflow.keras import Sequential, models, layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import kagglehub

# Descargar dataset
path = kagglehub.dataset_download(
    "bhavikjikadara/dog-and-cat-classification-dataset")

print("Path to dataset files:", path)

data = Path(path)
print("Datos disponibles:", data)

# Verificar archivos disponibles
image_files = list(data.glob('**/*.jpg'))
print(f"Total de imágenes encontradas: {len(image_files)}")
print("Ejemplos de rutas:", image_files[:5])

# Mostrar ejemplos de imágenes
try:
    cat_example = list(data.glob('PetImages/Cat/*.*'))[100]
    dog_example = list(data.glob('PetImages/Dog/*.*'))[100]
    print(f"Ejemplo de gato: {cat_example}")
    print(f"Ejemplo de perro: {dog_example}")

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Ejemplo de Gato")
    plt.imshow(np.array(Image.open(str(cat_example))))

    plt.subplot(1, 2, 2)
    plt.title("Ejemplo de Perro")
    plt.imshow(np.array(Image.open(str(dog_example))))
    plt.show()
except Exception as e:
    print(f"Error al mostrar imágenes de ejemplo: {e}")

# Preparar los datos
animals = {
    'dog': list(data.glob('PetImages/Dog/*.*')),
    'cat': list(data.glob('PetImages/Cat/*.*')),
}

label = {
    'dog': 1,
    'cat': 0,
}

# Preprocesamiento de imágenes


def load_and_preprocess_image(image_path, target_size=(150, 150)):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None

        # Convertir de BGR a RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Redimensionar
        img_resized = cv2.resize(img, target_size)

        # Normalizar
        img_normalized = img_resized / 255.0

        return img_normalized
    except Exception as e:
        print(f"Error procesando {image_path}: {e}")
        return None


# Cargar imágenes con barra de progreso
X = []
Y = []
total_images = sum(len(images) for images in animals.values())
processed = 0
print(f"Procesando {total_images} imágenes...")

for name, images in animals.items():
    for image in images:
        processed += 1
        if processed % 1000 == 0:
            print(f"Procesadas {processed}/{total_images} imágenes")

        image_path = str(image).strip()
        img_processed = load_and_preprocess_image(image_path)

        if img_processed is not None:
            X.append(img_processed)
            Y.append(label[name])

print(f"Imágenes cargadas correctamente: {len(X)} de {total_images}")

# Convertir a arrays de NumPy
X = np.array(X)
Y = np.array(Y)

print(f"Forma del conjunto de datos: {X.shape}")
print(f"Distribución de clases: {np.bincount(Y)}")

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y)

print(f"Conjunto de entrenamiento: {X_train.shape}, {y_train.shape}")
print(f"Conjunto de prueba: {X_test.shape}, {y_test.shape}")

# Aumentación de datos
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
])

# Construir un modelo más adecuado


def build_model(input_shape=(150, 150, 3)):
    model = Sequential([
        # Capa de aumentación de datos
        data_augmentation,

        # Primera capa convolucional
        layers.Conv2D(32, (3, 3), activation='relu',
                      padding='same', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),

        # Segunda capa convolucional
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        # Tercera capa convolucional
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        # Cuarta capa convolucional
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        # Aplanar y capas densas
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    # Compilar modelo
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


# Crear y mostrar resumen del modelo
model = build_model(input_shape=X_train[0].shape)
model.summary()

# Callbacks para mejorar el entrenamiento
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True,
                  monitor='val_accuracy'),
    ReduceLROnPlateau(factor=0.2, patience=5, min_lr=1e-6, monitor='val_loss')
]

# Entrenar modelo
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=5,
    batch_size=32,
    callbacks=callbacks
)

# Visualizar curvas de aprendizaje
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='validation')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.title('Loss')
plt.legend()
plt.show()

# Evaluar el modelo
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Precisión en el conjunto de prueba: {test_acc:.4f}")

# Guardar el modelo
model.save('models/cat_dog_classifier_improved.h5')
print("Modelo guardado correctamente")

# Realizar algunas predicciones


def predict_image(image_path, model):
    img = load_and_preprocess_image(image_path)
    if img is None:
        return "No se pudo cargar la imagen"

    # Agregar dimensión de batch
    img = np.expand_dims(img, axis=0)

    # Predecir
    prediction = model.predict(img)[0][0]
    class_name = "Perro" if prediction > 0.5 else "Gato"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    return f"Clase: {class_name}, Confianza: {confidence:.2f}"


# Probar predicciones con algunas imágenes
test_images = image_files[:5]
for img_path in test_images:
    result = predict_image(str(img_path), model)
    print(f"Imagen: {img_path}, Predicción: {result}")
