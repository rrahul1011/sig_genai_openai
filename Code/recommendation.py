import pandas as pd
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy

df = pd.read_csv("Data/df_final_with_name2.csv")

# Create a Surprise Dataset
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['user_id', 'product_id', 'rating']], reader)

# Split the dataset into training and testing sets
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Build and train the user-based collaborative filtering model (KNN)
sim_options = {
    'name': 'cosine',  
    'user_based': True  
}
model_cf = KNNBasic(sim_options=sim_options)
model_cf.fit(trainset)

# Make predictions with the collaborative filtering model
predictions_cf = model_cf.test(testset)

# Evaluate the collaborative filtering model's performance (e.g., RMSE)
rmse_cf = accuracy.rmse(predictions_cf)

# Recommend products for a specific user using collaborative filtering
def recommend_products_cf(user_id, num_recommendations=5):
    products_rated_by_user = df[df['user_id'] == user_id]['product_id']
    products_not_rated_by_user = df[~df['product_id'].isin(products_rated_by_user)]['product_id'].unique()
    
    recommendations = []
    
    for product_id in products_not_rated_by_user:
        predicted_rating = model_cf.predict(user_id, product_id).est
        recommendations.append((product_id, predicted_rating))
    
    recommendations.sort(key=lambda x: x[1], reverse=True)
    top_recommendations = recommendations[:num_recommendations]
    
    return [product[0] for product in top_recommendations]

# Content-based filtering function (replace with your own content-based model)
def content_based_recommendations(user_id, num_recommendations=5):
    user_genre_preferences = df[df['user_id'] == user_id]['product_genre'].value_counts().index
    user_genre_preferences = user_genre_preferences[:num_recommendations]  # Example: Recommend top genres
    
    content_based_recommendations = df[df['product_genre'].isin(user_genre_preferences)]['product_id'].unique()
    
    return content_based_recommendations

# Hybrid recommendation function
def hybrid_recommendations(user_id, num_recommendations=5):
    cf_recommendations = recommend_products_cf(user_id, num_recommendations)
    content_based_rec = content_based_recommendations(user_id, num_recommendations)
    
    hybrid_rec = list(set(cf_recommendations) | set(content_based_rec))  # Union of recommendations
    
    return hybrid_rec[:num_recommendations]

# Example usage:
user_id = 3  # Replace with the desired user ID
hybrid_recommendations_list = hybrid_recommendations(user_id)
print("Hybrid Recommended products for User", user_id, ":")
print(hybrid_recommendations_list)
