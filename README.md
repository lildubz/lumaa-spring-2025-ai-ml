# Lyrics-Based Song Recommendation System

## Overview

This Python application provides song recommendations based on lyric similarity. It uses natural language processing techniques (TF-IDF) to find songs with lyrics that match the user's description of their music preferences. Users can further refine results by applying optional filters for genre and release year.

## Features

- Text-based song recommendations using lyrics similarity
- Optional filtering by genre and release year
- Interactive command-line interface
- Dataset size reduction for improved performance
- Similarity score display for each recommendation

## Requirements

- Python 3.6+
- pandas
- scikit-learn
- numpy

## Installation

1. Clone this repository or download the script
2. Install the required packages:

```bash
pip install pandas scikit-learn numpy
```

3. Prepare your song dataset in CSV format with these required columns:
   - `track_name`: The name of the song
   - `artist_name`: The name of the artist
   - `genre`: The genre of the song
   - `release_date`: The year the song was released
   - `lyrics`: The lyrics of the song

## Usage

1. Run the script:

```bash
python song_recommendation.py
```

2. When prompted, provide the path to your CSV dataset file
3. Enter your music preferences when asked
4. Optionally, specify genre and release year filters
5. Review the recommendations with their similarity scores
6. Continue entering new preferences or type 'exit', 'quit', or 'done' to end the session

## How It Works

1. **Data Loading**: The system loads a CSV file containing song information and lyrics.
2. **Feature Engineering**: It creates a text feature column from song lyrics.
3. **TF-IDF Vectorization**: The lyrics are converted into numerical vectors using TF-IDF (Term Frequency-Inverse Document Frequency).
4. **Similarity Calculation**: When a user enters their preferences, the system calculates the cosine similarity between the user input and all songs in the dataset.
5. **Filtering**: Optional genre and year filters are applied to narrow down the results.
6. **Recommendation**: The top N songs with the highest similarity scores are displayed.

## Sample Interaction

```
Please enter the file path of the dataset: /path/to/your/dataset.csv
Dataset loaded successfully!

Please describe your music preferences: upbeat songs about summer and love
Enter a genre to filter by (or press Enter to skip): pop
Enter a release year to filter by (or press Enter to skip): 2020

User Input: upbeat songs about summer and love

Recommended Songs:

Rank: 1 | Song: Summer Love | Artist: Justin Timberlake | Genre: pop | Year: 2020 
Rank: 2 | Song: Hot Summer Nights | Artist: Miami Sound Machine | Genre: pop | Year: 2020 
...
```

## Limitations

- The quality of recommendations depends on the availability and quality of lyrics in the dataset
- Large datasets may require significant processing time
- The system uses only lyrics for similarity and doesn't consider audio features
- The dataset is reduced to 500 entries for performance optimization

## Future Improvements

- Implement audio feature-based recommendations
- Add support for more complex filtering options
- Create a web or GUI interface
- Optimize for larger datasets
- Add support for collaborative filtering

