import os
from sklearn.metrics.pairwise import cosine_similarity
from .utils import extract_feature, get_transform

def build_feature_database(dataset_folder, feature_extractor):
    transform = get_transform()
    feature_db = {}
    for file in os.listdir(dataset_folder):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(dataset_folder, file)
            feature_db[image_path] = extract_feature(image_path, feature_extractor, transform)
            print(f"Processed dataset image: {file}")
    return feature_db

def find_similar_images(query_feature, feature_db, top_k=5):
    similarities = []
    for image_path, feature in feature_db.items():
        sim = cosine_similarity([query_feature], [feature])[0][0]
        similarities.append((image_path, sim))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]
