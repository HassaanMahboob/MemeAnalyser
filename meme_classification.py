# %%
import pandas as pd
import numpy as np
from PIL import Image
import re
import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize 
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.style as style
import ast
from sklearn.preprocessing import MultiLabelBinarizer
from string import punctuation
from gensim.models import KeyedVectors
import pytesseract

import pickle

# %%
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense, Input, Bidirectional, Flatten, Conv2D, MaxPooling2D, concatenate, Conv1D, MaxPooling1D
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam, RMSprop
from  tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
import tensorflow as tf
from utils import clean_text

def init_model():
    model = tf.keras.models.load_model("Models/hatememe_final_model.h5")
    tokenizer = pickle.load(open("Models/tokenizer_text.pk", "rb" ))
    return model, tokenizer


from keras.preprocessing.text import text_to_word_sequence


def is_hateful(img_path, model, tokenizer, MAX_SEQUENCE_LENGTH=1601):
    text = OCR_img2txt(img_path)
    text = clean_text (text)
    image = Image.open(img_path)
    
    
    image= image.resize((200,200))
    image = np.asarray(image)
    image = np.array([image])
    text = tokenizer.texts_to_sequences(text)
    temp_text = list()
    for t in text:
        try:
            temp_text.append(t[0])
        except:
            pass
    text = temp_text
    text = np.array([text])
    text = pad_sequences(text, maxlen=MAX_SEQUENCE_LENGTH)
    return model.predict([np.array(image),text])


def OCR_img2txt(img_path):
    pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
    return pytesseract.image_to_string(Image.open(img_path))
    

if __name__ == "__main__":

    classifier_model, classifier_tokenizer = init_model()
    result = is_hateful("test.png", classifier_model, classifier_tokenizer)
    print(result)


