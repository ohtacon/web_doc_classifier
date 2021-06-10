import string, os
import docx
import re
import random


# ПОЛУЧЕНИЕ СПИСКА ДОКУМЕНТОВ ПО ПУТИ К ПАПКЕ
def get_docs(dir_path):
    filenames = os.listdir(dir_path)
    docs = []
    for filename in filenames:
        if filename[-5:] == '.docx':
            doc_path = os.path.join(dir_path, filename)
            doc = docx.Document(doc_path)
            docs.append(doc)
    return docs


# Заявления
docs0_dir = r'C:\Users\Артём\Desktop\Python\Заявления'
docs0 = get_docs(docs0_dir)  # 150 docs
# Другие документы
docs1_dir = r'C:\Users\Артём\Desktop\Python\Другие документы'
docs1 = get_docs(docs1_dir)  # 160 docs
# Агентские договоры
docs2_dir = r'C:\Users\Артём\Desktop\Python\Агентские договоры'
docs2 = get_docs(docs2_dir)  # 76 docs
# Договоры купли-продажи
docs3_dir = r'C:\Users\Артём\Desktop\Python\Договоры купли-продажи'
docs3 = get_docs(docs3_dir)  # 100 docs
# Договоры аренды
docs4_dir = r'C:\Users\Артём\Desktop\Python\Договоры аренды'
docs4 = get_docs(docs4_dir)  # 149 docs
# Договоры услуги
docs5_dir = r'C:\Users\Артём\Desktop\Python\Договоры услуги'
docs5 = get_docs(docs5_dir)  # 221 docs

doc_lists = [docs0, docs1, docs2, docs3, docs4, docs5]
# ДАЛЕЕ ПОЛУЧЕННЫЕ СПИСКИ ДОКУМЕНТОВ РАЗБИВАЮТСЯ НА ЧАСТИ СОГЛАСНО РАСПРЕДЕЛЕНИЮ
docs_count = 0
for doc_list in doc_lists:
    docs_count += len(doc_list)

maxlen = 1024
max_words = 10000
training_samples = round(0.6 * docs_count)
validation_samples = round(0.2 * docs_count)
test_samples = docs_count - training_samples - validation_samples


train_part = training_samples / docs_count
val_part = validation_samples / docs_count
test_part = 1 - train_part - val_part

import math

# МЕТОД ВЗОВРАЩАЕТ СПИСОК МЕТОК СОЗДАННЫЙ ИЗ ДОЛЕЙ СПИСКОВ ДОКУМЕНТОВ (ДОЛЯ==part)
# ДОЛЯ МОЖЕТ НАХОДИТЬСЯ В ПРЕДЕЛАХ [0, 1]
def get_parted_list_of_labels(doc_lists, part=1.0):
    if part < 0 or part > 1:
        raise Exception('ДОЛЯ ДОЛЖНА БЫТЬ В ПРЕДЕЛАХ [0, 1]')
    parted_labels = []
    for label, doc_list in enumerate(doc_lists):
        docs_count = math.floor(part * len(doc_list))
        parted_labels += [label for i in range(docs_count)]
    return parted_labels


ytrain = get_parted_list_of_labels(doc_lists, train_part)
yval = get_parted_list_of_labels(doc_lists, val_part)
ytest = get_parted_list_of_labels(doc_lists, test_part)
print(ytrain, '\n', yval, '\n', ytest)
print(len(ytrain), len(yval), len(ytest))
print(len(ytrain) + len(yval) + len(ytest))
print(docs_count)

# ИЗ-ЗА ПРОЦЕНТНЫХ СООТНОШЕНИЙ НЕМНОГО ИЗМЕНИЛОСЬ ЧИСЛО ОБРАЗЦОВ TRAINING_SAMPLES, VALIDATION_SAMPLES, TEST_SAMPLES
# ПОЭТОМУ ИМ НУЖНО ПРИСВОИТЬ ИЗМЕНИВШИЕСЯ ЗНАЧЕНИЯ
training_samples = len(ytrain)
validation_samples = len(yval)
test_samples = len(ytest)
# СПИСКИ С МЕТКАМИ НУЖНО ПЕРЕМЕШАТЬ ПО ОТДЕЛЬНОСТИ, А НЕ ПОСЛЕ СОЕДИНЕНИЯ ЭТИХ СПИСКОВ В ЕДИНЫЙ СПИСОК,
# ЧТОБЫ НЕ НАРУШИТЬ ПРОЦЕНТНОЕ СОДЕРЖАНИЕ ТИПОВ ДОКУМЕНТОВ В ВЫБОРКАХ
# random.seed(5)
random.shuffle(ytrain)
random.shuffle(yval)
random.shuffle(ytest)
all_labels = ytrain + yval + ytest

print(all_labels)

# СОГЛАСНО СПИСКУ МЕТОК СОСТАВЛЯЕТСЯ СООТВУТСТВУЮЩИЙ СПИСОК ДОКУМЕНТОВ
def get_docs_by_labels(labels):
    all_docs = []
    for label in labels:
        all_docs.append(doc_lists[label].pop())
    return all_docs



all_docs = get_docs_by_labels(all_labels)
# ПОМЕСТИМ КОНТРОЛЬНЫЕ ДОКУМЕНТЫ В ОТДЕЛЬНУЮ ПАПКУ
# dir_test = r'C:\Users\Артём\Desktop\Python\test_set'
# test_labels = all_labels[training_samples + validation_samples: training_samples + validation_samples + test_samples]
# TODO

# ПОЛУЧЕНИЕ СПИСКА ТЕКСТОВ ИЗ СПИСКА ДОКУМЕНТОВ
def get_texts(docs):
    texts = []
    for doc in docs:
        text = ''
        for par in doc.paragraphs:
            text += f'{par.text}\n'
        texts.append(text)
    return texts


docs_texts = get_texts(all_docs)
# ИЗБАВЛЕНИЕ ОТ ЗНАКОВ ПРЕПИНАНИЯ В ПОЛУЧЕННЫХ ТЕКСТАХ
pure_texts = []
for doc_text in docs_texts:
    text = re.sub(f'[{string.punctuation}]', '', doc_text)
    pure_texts.append(text)
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

# CONVERTING TEXTS TO SEQUENCES USING KERAS TOKENIZER
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(pure_texts)
sequences = tokenizer.texts_to_sequences(pure_texts)
# SAVING TOKENIZER
import pickle

with open('tokenizer_v3.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

data = pad_sequences(sequences, maxlen=maxlen)

labels = np.asarray(all_labels)

print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]
x_test = data[training_samples + validation_samples: training_samples + validation_samples + test_samples]
y_test = labels[training_samples + validation_samples: training_samples + validation_samples + test_samples]

# input()
from keras.models import Sequential
from keras import layers

model = Sequential()
model.add(layers.Embedding(max_words, 256, input_length=maxlen))
model.add(layers.Conv1D(64, 8, activation='relu'))
model.add(layers.MaxPooling1D(4))
model.add(layers.Conv1D(64, 8, activation='relu'))
model.add(layers.MaxPooling1D(4))
model.add(layers.Conv1D(64, 8, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(len(doc_lists), activation='softmax'))

from keras import losses
from keras import optimizers

model.compile(optimizer=optimizers.RMSprop(lr=1e-3),
              loss=losses.sparse_categorical_crossentropy,
              metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=15,
                    batch_size=1,
                    validation_data=(x_val, y_val))

model.save('pretrained_docs_model_v3.h5')

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Точность на этапе обучения')
plt.plot(epochs, val_acc, 'b', label='Точность на этапе проверки')
plt.title('Точность обучения и проверки')
plt.legend()
plt.show()

prediction = model.evaluate(x_test, y_test, batch_size=1)
print(f'Размер контрольной выборки: {test_samples} документов ==> {test_samples / docs_count * 100}% от всех')
print("Потери и точность на контрольной выборке")
print(prediction)
