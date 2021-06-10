import string, os
import docx
import re
import random

# ПОЛУЧЕНИЕ СПИСКА ДОКУМЕНТОВ ПО ПУТИ К ПАПКЕ
def get_docs(dir_path):
    file_names = os.listdir(dir_path)
    docs = []
    for filename in file_names:
        if filename[-5:] == '.docx':
            doc_path = os.path.join(dir_path, filename)
            doc = docx.Document(doc_path)
            docs.append(doc)
    return docs


# Заявления
docs0_dir = r'C:\Users\Артём\Desktop\Python\Заявления'
docs0 = get_docs(docs0_dir)
# Доверенности
docs1_dir = r'C:\Users\Артём\Desktop\Python\Доверенности'
docs1 = get_docs(docs1_dir)


# Перемешивается список меток и в соответствии с ним создаётся список документов
all_labels = [0 for i in docs0] + [1 for i in docs1]
random.seed(1)
random.shuffle(all_labels)
all_docs = []
for label in all_labels:
    if label == 0:
        all_docs.append(docs0.pop())
    if label == 1:
        all_docs.append(docs1.pop())
# print(len(all_docs), len(all_labels))

# ПОЛУЧЕНИЕ СПИСКА ТЕКСТОВ ИЗ СПИСКА ДОКУМЕНТОВ
def get_texts(docs):
    texts = []
    for doc in docs:
        text = ''
        for par in doc.paragraphs:
            text += f'{par.text}\n'
        texts.append(text)
    return texts


# очистка полученных текстов от знаков препинания
docs_texts = get_texts(all_docs)
# ИЗБАВЛЕНИЕ ОТ ЗНАКОВ ПРЕПИНАНИЯ В ПОЛУЧЕННЫХ ТЕКСТАХ
pure_texts = []
for doc_text in docs_texts:
    text = re.sub(f'[{string.punctuation}]', '', doc_text)
    pure_texts.append(text)


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

maxlen = 300
training_samples = 120
validation_samples = 85
max_words = 5000

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(pure_texts)
sequences = tokenizer.texts_to_sequences(pure_texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=maxlen)

labels = np.asarray(all_labels)

print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]


# input()
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.Embedding(max_words, 128, input_length=maxlen))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.GlobalMaxPooling1D())
# model.add(layers.Flatten())
model.add(layers.Dense(1, activation='sigmoid'))

# model.add(layers.Conv1D(32, 7, activation='relu'))
# model.add(layers.GlobalMaxPooling1D())
# model.add(layers.Dense(1))
# print(model.summary())
# input()
from keras import losses
model.compile(optimizer=RMSprop(lr=1e-5),
              loss=losses.binary_crossentropy,
              metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=20,
                    batch_size=2,
                    validation_data=(x_val, y_val))

# model.save_weights('pretrained_docs_model.h5')

import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and Validation accuracy')
plt.legend()
plt.show()

# for text in pure_texts:
#     print(text)
#     print(('-' * 1000 + '\n') * 5)

# составляется список слов, содержащихся в документе docx
# word_sequences = []
# for i, text in enumerate(docs_texts):
#     word_sequences.append([])
#     for line in text:
#         word_sequences[i] += re.sub('[' + string.punctuation + ']', '', line).split()
# ПЕЧАТЬ ПОСЛЕДОВАТЕЛЬНОСТЕЙ ПОСЛЕДОВАТЕЛЬНОСТЕЙ
# for sequence in word_sequences:
#     print(sequence)
