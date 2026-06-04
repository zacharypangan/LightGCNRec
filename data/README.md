# Data Layout

Raw datasets and generated artifacts are not tracked in git. Download the public datasets separately and place prepared files under this directory using the layouts below.

## MovieLens-1M

Expected files:

```text
data/ml-1m/
  movies_augmented.csv
  ratings_filtered.csv
  recmov_ratings.csv
  proxmov_ratings.csv
  rec_prox_ratings.csv
  embeddings/
    diverse_user_item_text_embeddings.json
    diverse_user_item_text_embeddings_no_movies.json
```

Required interaction columns for package loaders:

- `userId`
- `movieId`
- `rating`

## Steam

Expected files:

```text
data/steam/
  games_cleaned.csv
  ratings_cleaned.csv
  users_cleaned.csv
  augonly_user_rec.csv
  enhanced_game_recommendations.csv
  augmented_user_recommendations.csv
  embeddings/
    user_item_text_embeddings.json
    user_item_text_embeddings_games.json
```

Required interaction columns for package loaders:

- `user_id`
- `app_id`
- rating or implicit interaction field, depending on the prepared split

## Generated Artifacts

Keep these paths local and untracked:

- `models/` for checkpoints
- `outputs/` for predictions and metrics
- `results/` for figures and tables
- `data/**/embeddings/` for generated LLM embedding caches

The configs in `configs/` document the expected path names and can be adjusted for your machine.
