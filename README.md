# Recommendation System

# Recommendation System

This project implements a collaborative filtering recommendation system using user ratings of products. The goal is to suggest products to users based on the preferences and ratings of similar users.

## How It Works

1. **Data Collection**: The system uses a dataset of user ratings for various products, where each entry consists of a user ID, product ID, and the rating given by the user.

2. **User-Item Matrix**: The first step in the recommendation process involves creating a user-item matrix from the ratings data. This matrix represents users in rows and products in columns, with ratings filling in the corresponding cells. Users who have not rated a product will have a value of zero in that cell.

3. **Similarity Calculation**: The recommendation system calculates the similarity between users using cosine similarity. This metric evaluates how similar two users are based on their ratings. Users who have rated products similarly will have a higher cosine similarity score.

4. **Predicting Ratings**: To recommend products, the system predicts a user's rating for a product they haven't seen yet. This is done by looking at the ratings given by similar users and calculating a weighted average based on their similarity scores.

5. **Generating Recommendations**: Finally, the system generates a list of recommended products for the specified user. The recommendations are sorted based on the predicted ratings, allowing users to discover new products that they are likely to enjoy.

## Benefits of Collaborative Filtering

- **Personalized Recommendations**: The system provides personalized suggestions based on individual user preferences.
- **No Need for product Metadata**: Unlike content-based filtering, collaborative filtering does not require additional information about the products, making it adaptable to various datasets.
- **Diverse Recommendations**: By leveraging the ratings of similar users, the system can suggest a wide variety of products, enhancing user engagement.

## Use Cases

- **Streaming Services**: This system can be integrated into platforms like Netflix or Hulu to enhance user experience by suggesting products and shows tailored to individual tastes.
- **E-commerce**: Online retailers can implement similar recommendation systems to suggest products based on user ratings and preferences.
- **Social Media**: Platforms can recommend content or connections based on user interactions and preferences.

By utilizing collaborative filtering techniques, this project aims to provide an effective and scalable solution for product recommendations.

## Project Structure

```
.
├── data
│   └── final_filtered_ratings.csv  # Input ratings data
├── recommend.py                    # Main file to run the API and get recommendations
└── requirements.txt                # Python dependencies
```

## Usage

1. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the recommendation system:

   ```bash
   python recommend.py
   ```

3. Get recommendations for a specific user:
   ```bash
   curl -X GET "http://127.0.0.1:5000/recommend?user_id=<USER_ID>"
   ```

Replace `<USER_ID>` with the desired user's ID.

## Requirements

- Python 3.x
- pandas
- scikit-learn
- Flask
