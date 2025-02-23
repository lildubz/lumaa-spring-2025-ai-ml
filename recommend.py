import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Function to load the dataset
def load_dataset():
    file_path = input("Please enter the file path of the dataset (mine -> /Users/ldubin/Desktop/internship/tcc_ceds_music.csv): ")
    try:
        df = pd.read_csv(file_path)
        #reduce dataset to 500 rows by evenly sampling
        df_reduced = df.iloc[::len(df) // 500][:500]
        df_reduced.to_csv("songs_reduced.csv", index = False)
        print("Dataset loaded successfully!")
        return df
    except FileNotFoundError:
        print("Error: File not found. Please check the file path and try again.")
        return None

# Create a combined text feature column using song lyrics
def create_text_features(df):
    df["text_features"] = df["lyrics"].fillna("")  # Using lyrics as the feature for comparison
    return df

# Initialize TF-IDF Vectorizer and compute the matrix
def compute_tfidf_matrix(df):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df["text_features"])
    return vectorizer, tfidf_matrix

def recommend_songs(user_input, df, vectorizer, tfidf_matrix, top_n=5, genre_filter=None, year_filter=None):
    """
    Recommend the top N most similar songs based on lyrics while allowing optional filters.
    """
    # Transform the user input into a TF-IDF vector
    user_vector = vectorizer.transform([user_input])

    # Compute cosine similarity between user input and dataset songs
    similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()

    # Get indices sorted by highest similarity
    sorted_indices = np.argsort(similarities)[::-1]

    # Convert dataset to a DataFrame and add similarity scores
    recommendations = df.iloc[sorted_indices].copy()
    recommendations["similarity_score"] = similarities[sorted_indices]
    
    # Apply optional filters
    if genre_filter:
        recommendations = recommendations[recommendations["genre"].str.lower() == genre_filter.lower()]
    if year_filter:
        recommendations = recommendations[recommendations["release_date"] == year_filter]

    # Select the top N recommendations
    recommendations = recommendations.head(top_n).reset_index(drop=True)

    # Print recommendations without similarity scores
    print(f"\nUser Input: {user_input}\n")
    print("Recommended Songs:\n")
    for index, row in recommendations.iterrows():
        print(f"Rank: {index+1} | Song: {row['track_name']} | Artist: {row['artist_name']} | Genre: {row['genre']} | Year: {row['release_date']}")

    return recommendations


def get_user_preferences():
    """
    Prompt the user for music preferences including optional filters.
    """
    user_input = input("Please describe your music preferences: ")
    genre_filter = input("Enter a genre to filter by (or press Enter to skip): ").strip()
    year_filter = input("Enter a release year to filter by (or press Enter to skip): ").strip()
    
    # Convert year_filter to an integer if provided
    year_filter = int(year_filter) if year_filter.isdigit() else None

    return user_input, genre_filter if genre_filter else None, year_filter

def main():
    """
    Main loop that allows the user to keep asking for song recommendations.
    """
    # Load dataset
    df = load_dataset()
    if df is None:
        return

    # Create text features from lyrics and compute TF-IDF matrix
    df = create_text_features(df)
    vectorizer, tfidf_matrix = compute_tfidf_matrix(df)

    # Main loop for user interaction
    while True:
        # Get user input and filters
        user_input, genre_filter, year_filter = get_user_preferences()

        # If user wants to exit
        if user_input.lower() in ['exit', 'quit', 'done']:
            print("Exiting the recommendation system. Goodbye!")
            break

        # Get recommendations based on user input and filters
        recommend_songs(user_input, df, vectorizer, tfidf_matrix, genre_filter=genre_filter, year_filter=year_filter)

if __name__ == "__main__":
    main()
