# Indie Game Publisher Recommender

Content-based recommender that helps indie developers identify potential publishers by matching a game concept to publishers with a similar catalog history.

**[Live app](https://indie-publisher-search.streamlit.app/)**

## Overview

The app takes a set of Steam tags and a price point, then returns publishers whose existing catalogs are the closest match, along with concrete examples of titles they've published. The goal is to turn "publisher fit" into something quantifiable, so developers can approach publishers with a track record in their genre instead of cold-emailing at random.

## Features

- **Tag vectorization**: TF-IDF over each game's Steam tags, capturing genre, sub-genre, and mechanic overlap
- **Price alignment**: price is scaled and blended into the similarity score (with an adjustable weight), so matches also reflect commercial positioning, not just genre
- **Evidence-based output**: every recommended publisher comes with real example titles and Steam links, not just a score
- **Tunable controls**: sliders for price influence, number of publishers returned, and minimum catalog size (filters out one-off publishers)

## Repo Structure

```
├── data/                   # pixel_games_sales_pairs.csv: game-publisher pairs with tags, price, Steam metadata
├── .notebook/              # Data preparation and exploration
├── 02_streamlit_app.py     # Streamlit app: TF-IDF + price model, similarity search, UI
├── requirements.txt
└── README.md
```

- **`02_streamlit_app.py`** is the full app: data loading, the TF-IDF and price model, publisher profile building, and the recommendation UI.
- **`.notebook/`** holds the data preparation and exploration behind `pixel_games_sales_pairs.csv`.

## Methods

| Step | Approach |
|---|---|
| Tag vectorization | TF-IDF over each game's Steam tags |
| Price feature | Price scaled (StandardScaler) and weighted, then combined with the tag vector |
| Publisher profiles | Mean feature vector across each publisher's catalog, publishers below a minimum game count excluded |
| Recommendation | Cosine similarity between the query (tags and price) and each publisher profile |
| Examples | Cosine similarity within a recommended publisher's own catalog surfaces its closest-matching titles |

## Tools

Python · `pandas` / `numpy` · `scikit-learn` (TfidfVectorizer, StandardScaler, cosine_similarity) · `scipy.sparse` · `streamlit`

## Reproducing

1. Place `pixel_games_sales_pairs.csv` in `data/`.
2. Install required packages: `pip install -r requirements.txt`
3. Run the app: `streamlit run 02_streamlit_app.py`

## Data

`data/pixel_games_sales_pairs.csv` holds one row per game, with `tags` (list of Steam tags), `price`, `publisher`, `name`, `steamId`, and `steamUrl`. Sourced from a public dataset of 6,422 Steam games tagged Pixel Graphics, released between August 2023 and August 2025, originally compiled from Gamalytic data and shared via [Google Sheet](https://docs.google.com/spreadsheets/d/1I9_YeZoTRc7bRSlbVXNPX9SQWSrseIVmXzBGbQtyExg/edit?usp=sharing) in this [r/gamedev post](https://www.reddit.com/r/gamedev/comments/1n998gw/i_pulled_data_on_6422_pixel_art_games_released/). The sheet may have been updated since this project pulled from it.
