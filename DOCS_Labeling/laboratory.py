import fitz, docx


def extract_text(filepath):
    if filepath[-4:] == '.pdf':
        doc = fitz.open(filepath)
        text = ''
        for page in doc:
            t = page.getText()
            text += t
        return text
    elif filepath[-5:] == '.docx':
        doc = docx.Document(filepath)
        text = ''
        for par in doc.paragraphs:
            text += par.text + '\n'
        return text
    else:
        raise Exception('ФОРМАТ ФАЙЛА НЕ ПОДДЕРЖИВАЕТСЯ')


import re, string


def purify_texts(texts):
    pure_texts = []
    for text in texts:
        pure_texts.append(re.sub(f'[{string.punctuation}]', '', text))
    return pure_texts


import os

base_dir = r"C:\Users\Артём\Desktop\Python"

# Создаётся список путей к файлам
check_dir = os.path.join(base_dir, 'test_set')
check_dir_filenames = []
for filename in os.listdir(check_dir):
    filepath = os.path.join(check_dir, filename)
    if os.path.isfile(filepath):
        check_dir_filenames.append(filepath)


def get_texts(filenames):
    texts = []
    for filename in filenames:
        texts.append(extract_text(filename))
    return texts


# LOADING TOKENIZER
import pickle

with open('tokenizer_v4.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# GETTING DATA IN THE RIGHT FORMAT WITH TOKENIZER
texts = get_texts(check_dir_filenames)
pure_texts = purify_texts(texts)
sequences = tokenizer.texts_to_sequences(pure_texts)

from keras.preprocessing.sequence import pad_sequences

MAXLEN = 2048

data = pad_sequences(sequences, maxlen=MAXLEN)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# PREDICTING RESULTS
from keras import models

model = models.load_model('pretrained_docs_model_v4.h5')
pred = model.predict(data)
print(pred)

import numpy as np


def get_predicted_labels(prediction):
    predicted_labels = []
    for list_of_probabilities in prediction:
        index_of_max = np.argmax(list_of_probabilities)
        predicted_labels.append(index_of_max)
    return predicted_labels


label_names = ['Заявления', 'Другие документы', 'Агентские договоры', 'Договоры купли-продажи',
               'Договоры аренды', 'Договоры услуги', "Доверенности"]
# label_names = ['Заявления', 'Другие документы', 'Агентские договоры', 'Договоры купли-продажи',
#                'Договоры аренды', 'Договоры услуги', "Доверенности", "Договоры подряда", "Договоры поставки",
#                "Договоры займа", "Трудовые договоры", "Уставы", "Приказы", "Договоры цессии", "Брачные договоры",
#                "Договоры найма", "Завещания", "Претензии", "Договоры лизинга", "Ходатайства"]


predicted_labels = get_predicted_labels(pred)
print(predicted_labels)

import shutil

for label_name in label_names:
    if not os.path.exists(os.path.join(check_dir, label_name)):
        os.mkdir(os.path.join(check_dir, label_name))

for index, label in enumerate(predicted_labels):
    dst_dir = os.path.join(check_dir, label_names[label])
    shutil.copy(check_dir_filenames[index], dst_dir)


print('КОПИРОВАНИЕ ФАЙЛОВ ЗАВЕРШЕНО')
