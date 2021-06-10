from flask import Flask, jsonify
from flask import url_for
# FLASK QUICKSTART

app = Flask(__name__)


@app.route('/')
def root():
    return 'ROOT'


@app.route('/hello', methods=['GET'])
def hello():
    urls = {}
    urls['root'] = url_for('root', _external=True)
    urls['hello'] = url_for('hello', _external=True)
    urls['show_user_profile'] = url_for('show_user_profile', username='<USERNAME>', _external=True)
    urls['show_post'] = url_for('show_post', post_id=1, _external=True)
    urls['show_subpath'] = url_for('show_subpath', subpath='[SUBPATH]', _external=True)
    urls['style'] = url_for('static', filename='style.css')
    return jsonify(urls)


from markupsafe import escape


@app.route('/user/<username>')
def show_user_profile(username):
    # show the user profile for that user
    return 'User %s' % escape(username)


@app.route('/post/<int:post_id>')
def show_post(post_id):
    # show the post with the given id, the id is an integer
    return 'Post %d' % post_id


@app.route('/path/<path:subpath>')
def show_subpath(subpath):
    # show the subpath after /path/
    return 'Subpath %s' % escape(subpath)


if __name__ == '__main__':
    app.run(debug=True, host='192.168.43.3')
