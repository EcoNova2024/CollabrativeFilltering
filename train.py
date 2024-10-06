import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import mysql.connector

def load_data_from_db():
    """Load the ratings dataset from the MySQL database."""
    # Retrieve database connection details from environment variables
    db_host = os.getenv('DB_HOST')
    db_user = os.getenv('DB_USER')
    db_password = os.getenv('DB_PASSWORD')
    db_name = os.getenv('DB_NAME')
    db_port = os.getenv('DB_PORT')

    # Establish a connection to the database
    connection = mysql.connector.connect(
        host=db_host,  
        user=db_user,  
        password=db_password,  
        database=db_name,  
        port=db_port  
    )
    
    query = "SELECT user_id, product_id, score FROM ratings"
    # Load data into a DataFrame
    df = pd.read_sql(query, connection)
    connection.close()  # Close the database connection
    return df

def train_model(filtered_ratings_df):
    """Train the model by calculating user similarities."""
    # Step 1: Split the dataset into training and testing sets
    train_df, _ = train_test_split(filtered_ratings_df, test_size=0.2, random_state=42)

    # Step 2: Create a user-item matrix from the training data
    user_item_matrix_train = train_df.pivot(index='user_id', columns='product_id', values='score').fillna(0)

    # Step 3: Calculate similarity between users using cosine similarity
    user_similarity_df = calculate_user_similarity(user_item_matrix_train)

    # Save the user similarity DataFrame to a CSV file
    user_similarity_df.to_csv("data/user_similarity.csv")
    print("User similarity data saved successfully.")

def calculate_user_similarity(user_item_matrix):
    """Calculate cosine similarity between users."""
    user_similarity = cosine_similarity(user_item_matrix)
    return pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

if __name__ == '__main__':
    filtered_ratings_df = load_data_from_db()
    train_model(filtered_ratings_df)
