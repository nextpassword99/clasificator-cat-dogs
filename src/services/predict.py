from tensorflow.keras.models import load_model
import cv2
import numpy as np


class PredictService:
    def __init__(self, path_model):
        self.model = load_model(path_model)

    def start(self, image_path):
        if self.model is None:
            print('No se ha cargado el modelo')
            return None
        if image_path is None:
            print('Sube una imagen')
            return None

        process_image = self._preprocess_image(image_path)
        if process_image is None:
            print('Error al procesar la imagen')
            return None

        prediction = self._predict_image(process_image)
        return prediction

    def _preprocess_image(self, image_path, target_size=(150, 150)):
        try:
            image_path = image_path.replace('"', '').trip()
            img = cv2.imread(image_path)
            if img is None:
                print(f'No se pudo cargar la imagen: {image_path}')
                return None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img, target_size)
            img_normalized = img_resized / 255.0

            return img_normalized.astype(np.float32)
        except Exception as e:
            print(f'Error al procesar la imagen: {e}')
            return None

    def _predict_image(self, img):
        img = np.expand_dims(img, axis=0)
        return self.model.predict(img)[0][0]
