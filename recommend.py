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
    """Train the model by calculating user and item similarities."""
    train_df, _ = train_test_split(filtered_ratings_df, test_size=0.2, random_state=42)
    
    # User-Item Matrix
    user_item_matrix_train = train_df.pivot(index='user_id', columns='product_id', values='score').fillna(0)
    
    # User Similarity Matrix
    user_similarity_df = calculate_user_similarity(user_item_matrix_train)
    
    # Item Similarity Matrix
    item_similarity_df = calculate_item_similarity(user_item_matrix_train)
    
    # Save similarity matrices to CSV files in the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the script
    user_similarity_df.to_csv(os.path.join(script_dir, "data", "user_similarity.csv"))
    item_similarity_df.to_csv(os.path.join(script_dir, "data", "item_similarity.csv"))
    print("User and Item similarity data saved successfully.")

def calculate_user_similarity(user_item_matrix):
    """Calculate cosine similarity between users."""
    user_similarity = cosine_similarity(user_item_matrix)
    return pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

def calculate_item_similarity(user_item_matrix):
    """Calculate cosine similarity between items (products)."""
    item_similarity = cosine_similarity(user_item_matrix.T)
    return pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

def get_user_recommendations(user_id, top_n=20):
    """Generate item recommendations for a specific user based on user similarities."""
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the script
    user_similarity_df = pd.read_csv(os.path.join(script_dir, "data", "user_similarity.csv"), index_col=0)
    filtered_ratings_df = load_data_from_db()
    
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).head(top_n).index
    recommendations = filtered_ratings_df[filtered_ratings_df['user_id'].isin(similar_users)]
    
    product_scores = recommendations.groupby('product_id')['score'].mean()
    top_recommendations = product_scores.sort_values(ascending=False).head(top_n)

    return top_recommendations

def get_item_recommendations(product_id, top_n=20):
    """Generate recommendations for a product based on item-item similarities."""
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the script
    item_similarity_df = pd.read_csv(os.path.join(script_dir, "data", "item_similarity.csv"), index_col=0)
    
    # Get the most similar items to the given product
    similar_items = item_similarity_df[product_id].sort_values(ascending=False).head(top_n).index
    
    return similar_items

@app.route('/recommendations', methods=['GET'])
def recommend():
    """Get item recommendations for a specific user or item."""
    user_id = request.args.get('user_id')  # Fetch user_id as a string
    product_id = request.args.get('product_id')  # Fetch product_id if provided
    
    if user_id:
        # Generate recommendations based on user similarities
        try:
            filtered_ratings_df = load_data_from_db()
            train_model(filtered_ratings_df)
            recommendations = get_user_recommendations(user_id)

            recommendations_dict = recommendations.to_dict()

            return jsonify({"user_id": user_id, "recommendations": recommendations_dict}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    elif product_id:
        # Generate recommendations based on item similarities
        try:
            filtered_ratings_df = load_data_from_db()
            train_model(filtered_ratings_df)
            similar_items = get_item_recommendations(product_id)

            # Convert similar_items (list of product ids) to a map with similarity scores
            script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the script
            item_similarity_df = pd.read_csv(os.path.join(script_dir, "data", "item_similarity.csv"), index_col=0)
            similar_items_dict = {item: item_similarity_df[product_id][item] for item in similar_items}

            return jsonify({"product_id": product_id, "similar_items": similar_items_dict}), 200
        except Exception as e:
            return jsonify({"error": f"Error fetching item recommendations: {str(e)}"}), 500
    else:
        return jsonify({"error": "Either user_id or product_id is required."}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5001)
