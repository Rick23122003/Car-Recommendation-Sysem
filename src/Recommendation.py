import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from src.Data_Preprocessing import load_and_preprocess_data
from src.Feature_Engineering import feature_engineering

# Load data
df = load_and_preprocess_data()

# Feature engineering
car_names, X = feature_engineering(df)

# 🔥 Remove zero vectors
non_zero_rows = np.linalg.norm(X, axis=1) != 0
X = X[non_zero_rows]
car_names = car_names[non_zero_rows].reset_index(drop=True)

# Normalize
X_normalized = normalize(X, norm="l2")

# Similarity matrix
similarity_matrix = cosine_similarity(X_normalized)

def recommend_cars(car_name, top_n=5):

    matches = car_names[car_names.str.contains(car_name, case=False, na=False)]

    if matches.empty:
        return "Car not found in dataset."

    car_index = matches.index[0]

    similarity_scores = list(enumerate(similarity_matrix[car_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    top_indices = [i[0] for i in similarity_scores[1:]]
    recommendations = car_names.iloc[top_indices].drop_duplicates()

    return recommendations.head(top_n)

if __name__ == "__main__":
    print("Recommended cars:\n")
    print(recommend_cars("Innova"))