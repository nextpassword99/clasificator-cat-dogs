import requests
from src.services.predict import PredictService


path_model = 'models/cat_dog_classifier_improved20.h5'
predict_service = PredictService(path_model)


while True:
    image_path = input('Ruta de la imagen: ')

    prediction = predict_service.start(image_path)
    print(prediction)

    path_esp = 'http://192.168.18.250/led'

    params = {'predict': prediction}
    response = requests.get(path_esp, params=params)

    print(response.text)
