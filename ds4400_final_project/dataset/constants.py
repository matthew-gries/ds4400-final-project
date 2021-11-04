from pathlib import Path

DOWNLOAD_FOLDER = str(Path(__file__).parents[2])
DATASET_FOLDER = str(Path(DOWNLOAD_FOLDER) / "Data")
DATASET_SLUG = "andradaolteanu/gtzan-dataset-music-genre-classification"