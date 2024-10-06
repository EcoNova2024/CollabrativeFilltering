import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import mysql.connector
from flask import Flask, request, jsonify

app = Flask(__name__)

def load_data_from_db():
    """Load the ratings dataset from the MySQL database."""
    db_host = os.getenv('DB_HOST')
    db_user = os.getenv('DB_USER')
    db_password = os.getenv('DB_PASSWORD')
    db_name = os.getenv('DB_NAME')
    db_port = os.getenv('DB_PORT')

    connection = mysql.connector.connect(
        host=db_host,
        user=db_user,
        password=db_password,
        database=db_name,
        port=db_port
    )

    query = "SELECT user_id, product_id, score FROM ratings"
    df = pd.read_sql(query, connection)
    connection.close()
    return df

def train_model(filtered_ratings_df):
    """Train the model by calculating user similarities."""
    train_df, _ = train_test_split(filtered_ratings_df, test_size=0.2, random_state=42)
    user_item_matrix_train = train_df.pivot(index='user_id', columns='product_id', values='score').fillna(0)
    user_similarity_df = calculate_user_similarity(user_item_matrix_train)

    # Save the user similarity DataFrame to a CSV file
    user_similarity_df.to_csv("data/user_similarity.csv")
    print("User similarity data saved successfully.")

def calculate_user_similarity(user_item_matrix):
    """Calculate cosine similarity between users."""
    user_similarity = cosine_similarity(user_item_matrix)
    return pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

def get_user_recommendations(user_id, top_n=20):
    """Generate item recommendations for a specific user based on user similarities."""
    user_similarity_df = pd.read_csv("data/user_similarity.csv", index_col=0)
    filtered_ratings_df = load_data_from_db()
    
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).head(top_n).index
    recommendations = filtered_ratings_df[filtered_ratings_df['user_id'].isin(similar_users)]
    
    product_scores = recommendations.groupby('product_id')['score'].mean()
    top_recommendations = product_scores.sort_values(ascending=False).head(top_n)

    return top_recommendations

@app.route('/recommendations', methods=['GET'])
def recommend():
    """Get item recommendations for a specific user ID."""
    user_id = request.args.get('user_id')  # Fetch user_id as a string

    if user_id is None:
        return jsonify({"error": "User ID is required."}), 400

    try:
        filtered_ratings_df = load_data_from_db()
        train_model(filtered_ratings_df)

        recommendations = get_user_recommendations(user_id)

        recommendations_dict = recommendations.to_dict()

        return jsonify({"user_id": user_id, "recommendations": recommendations_dict}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
if __name__ == '__main__':
    app.run(debug=True,port=5001)
