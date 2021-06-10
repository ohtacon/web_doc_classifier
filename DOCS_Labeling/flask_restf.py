from flask import Flask, url_for, render_template, make_response, request, redirect, flash
from flask_restful import reqparse, abort, Api, Resource

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'docx', 'pdf'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('task')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Todo
# shows a single todo item and lets you delete a todo item
# class Todo(Resource):
#     def get(self, todo_id):
# abort_if_todo_doesnt_exist(todo_id)
# return TODOS[todo_id]
# 
# def delete(self, todo_id):
#     abort_if_todo_doesnt_exist(todo_id)
#     del TODOS[todo_id]
#     return '', 204
# 
# def put(self, todo_id):
#     args = parser.parse_args()
#     task = {'task': args['task']}
#     TODOS[todo_id] = task
#     return task, 201

import os


class UploadFile(Resource):
    def get(self):
        return make_response(render_template('upload_page.html'))

    def post(self):
        # args = parser.parse_args()
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
            return redirect(url_for('uploaded', filename=filename))


class Uploaded(Resource):
    def get(self, filename):
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        return make_response(get_doc_classes([str(filepath)])[0])


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


import pickle


def get_tokenizer(filename):
    with open(filename, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer


from keras.preprocessing.sequence import pad_sequences
from keras import models
import numpy as np

LABEL_NAMES = ['Заявления', 'Другие документы', 'Агентские договоры', 'Договоры купли-продажи',\
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


api.add_resource(UploadFile, '/')
api.add_resource(Uploaded, '/uploaded/<filename>')

if __name__ == '__main__':
    app.run(debug=True)
