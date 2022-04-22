from PIL import Image
import numpy as np


import tensorflow
from tensorflow import Graph, keras
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.python.keras.models import load_model



# Always clear session before start
def init_model():
    K.clear_session()
    tensorflow.keras.backend.clear_session()

    model = load_model('models/meme_identifier.h5')
    return model


# # Maximum Image Uploading size
# app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024



def predict_is_meme(file, model):
    CATEGORIES = ['Not Meme', 'Meme']
    class_label = ''

    try:
        img = image.load_img(file, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        prediction = model.predict(x)
        class_label = CATEGORIES[int(prediction[0][0])]
    except Exception as e:
        pass
        print("Exception---->", e)

    return class_label



# model = init_model()
# print(predict_is_meme("test.png", model))


