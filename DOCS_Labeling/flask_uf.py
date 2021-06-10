import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'docx', 'pdf'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file', filename=filename))
    return render_template('upload_page.html')


from flask import send_from_directory
from flask import jsonify

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return get_doc_classes([str(filepath)])[0]


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
        raise Exception(f'ФОРМАТ ФАЙЛА НЕ ПОДДЕРЖИВАЕТСЯ')


def get_texts(filepaths):
    texts = []
    for filepath in filepaths:
        texts.append(extract_text(filepath))
    return texts


import re, string


def purify_texts(texts):
    pure_texts = []
    for text in texts:
        pure_texts.append(re.sub(f'[{string.punctuation}]', '', text))
    return pure_texts


# LOADING TOKENIZER
import pickle


def get_tokenizer(filename):
    with open(filename, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer


from keras.preprocessing.sequence import pad_sequences
from keras import models
import numpy as np


LABEL_NAMES = ['Заявления', 'Другие документы', 'Агентские договоры', 'Договоры купли-продажи', \
               'Договоры аренды', 'Договоры услуги', 'Доверенности']


def get_predicted_labels(prediction):
    predicted_labels = []
    for list_of_probabilities in prediction:
        index_of_max = np.argmax(list_of_probabilities)
        predicted_labels.append(LABEL_NAMES[index_of_max])
    return predicted_labels


MAXLEN = 2048
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tokenizer = get_tokenizer('tokenizer_v4.pickle')
model = models.load_model('pretrained_docs_model_v4.h5')


def get_doc_classes(filepaths):
    texts = get_texts(filepaths)
    pure_texts = purify_texts(texts)
    sequences = tokenizer.texts_to_sequences(pure_texts)
    data = pad_sequences(sequences, maxlen=MAXLEN)
    pred = model.predict(data)
    predicted_labels = get_predicted_labels(pred)
    return predicted_labels


if __name__ == '__main__':
    app.run(debug=True)
