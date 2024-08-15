# app.py
import os
from flask import Flask, request, render_template, redirect, url_for
from cbir.models import load_feature_extractor
from cbir.utils import get_transform, extract_feature
from cbir.database import build_feature_database, find_similar_images

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['DATASET_FOLDER'] = 'static/dataset'

feature_extractor = load_feature_extractor()

feature_db = build_feature_database(app.config['DATASET_FOLDER'], feature_extractor)
transform = get_transform()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'query_image' not in request.files:
            return redirect(request.url)
        file = request.files['query_image']
        if file.filename == '':
            return redirect(request.url)
        if file:
            query_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(query_path)
            query_feature = extract_feature(query_path, feature_extractor, transform)
            similar_images = find_similar_images(query_feature, feature_db)
            results = [(os.path.basename(path), sim) for path, sim in similar_images]
            return render_template('results.html', query_image=file.filename, results=results)
    return render_template('index.html')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
