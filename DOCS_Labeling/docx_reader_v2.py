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
# Другие документы
docs2_dir = r'C:\Users\Артём\Desktop\Python\Другие документы'
docs2 = get_docs(docs2_dir)

# Перемешивается список меток и в соответствии с ним создаётся список документов
all_labels = [0 for i in docs0] + [1 for i in docs1] + [2 for i in docs2]
random.seed(5)
random.shuffle(all_labels)
# print(all_labels)
# input()
all_docs = []
for label in all_labels:
    if label == 0:
        all_docs.append(docs0.pop())
    if label == 1:
        all_docs.append(docs1.pop())
    if label == 2:
        all_docs.append(docs2.pop())
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
# print(len(pure_texts))
# input()
maxlen = 400
training_samples = 220
validation_samples = 100
test_samples = 45
max_words = 7000
# CONVERTING TEXTS TO SEQUENCES USING KERAS TOKENIZER
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(pure_texts)
sequences = tokenizer.texts_to_sequences(pure_texts)
# SAVING TOKENIZER
import pickle
with open('tokenizer_v2.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=maxlen)

labels = np.asarray(all_labels)

print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# from keras.utils import to_categorical
# data = to_categorical(data)
# labels = to_categorical(labels)

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]
x_test = data[training_samples + validation_samples: training_samples + validation_samples + test_samples]
y_test = labels[training_samples + validation_samples: training_samples + validation_samples + test_samples]

# input()
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.Embedding(max_words, 128, input_length=maxlen))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.MaxPooling1D(6))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.MaxPooling1D(3))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
# model.add(layers.Flatten())
model.add(layers.Dense(3, activation='softmax'))

# model.add(layers.Conv1D(32, 7, activation='relu'))
# model.add(layers.GlobalMaxPooling1D())
# model.add(layers.Dense(1))
# print(model.summary())
# input()
from keras import losses

model.compile(optimizer=RMSprop(lr=1e-2),
              loss=losses.sparse_categorical_crossentropy,
              metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=15,
                    batch_size=1,
                    validation_data=(x_val, y_val))

model.save('pretrained_docs_model_v2.h5')

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


prediction = model.evaluate(x_test, y_test, batch_size=1)
print(prediction)
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
