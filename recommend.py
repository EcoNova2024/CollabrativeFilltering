import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the filtered ratings dataset
filtered_ratings_df = pd.read_csv("data/final_filtered_ratings.csv")  

# Step 1: Split the dataset into training and testing sets
train_df, test_df = train_test_split(filtered_ratings_df, test_size=0.2, random_state=42)

# Step 2: Create a user-item matrix from the training data
user_item_matrix_train = train_df.pivot(index='userId', columns='productId', values='rating').fillna(0)

# Step 3: Calculate similarity between users using cosine similarity
def calculate_user_similarity(user_item_matrix):
    """Calculate cosine similarity between users."""
    user_similarity = cosine_similarity(user_item_matrix)
    return pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

# Calculate user similarity for the training data
user_similarity_df = calculate_user_similarity(user_item_matrix_train)

# Save the user similarity DataFrame to a CSV file
user_similarity_df.to_csv("data/user_similarity.csv")
print("User similarity data saved successfully.")

# Function to get recommendations
def get_recommendations(user_id, user_similarity_df, user_item_matrix_train, top_n=20):
    if user_id not in user_similarity_df.index:
        return None  # Return None if user_id is not in the similarity DataFrame

    # Get similarity scores and user's ratings
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)
    user_ratings = user_item_matrix_train.loc[user_id]

    # Create a recommendation score for each item
    recommendation_scores = pd.Series(0, index=user_item_matrix_train.columns)

    for similar_user, similarity_score in similar_users.items():
        if similar_user != user_id:
            # Add weighted ratings of similar users
            recommendation_scores += similarity_score * user_item_matrix_train.loc[similar_user]

    # Remove items already rated by the user
    recommendation_scores = recommendation_scores[user_ratings == 0]

    # Sort recommendations by score and return the top N items
    return recommendation_scores.sort_values(ascending=False).head(top_n)

# API endpoint for recommendations
@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = request.args.get('user_id', type=int)  # Expect user_id as query parameter
    recommendations = get_recommendations(user_id, user_similarity_df, user_item_matrix_train, top_n=20)

    if recommendations is not None and not recommendations.empty:
        return jsonify(recommendations.to_dict()), 200  # Return recommendations as JSON
    else:
        return jsonify({"message": f"No recommendations available for User {user_id}."}), 404

if __name__ == '__main__':
    app.run(debug=True)
