from typing import List, Dict
from pathlib import Path
import csv

from ds4400_final_project.dataset.constants import DATASET_FOLDER


FEATURES_3_SEC_CSV = "features_3_sec.csv"
FEATURES_30_SEC_CSV = "features_30_sec.csv"
GENRES_ORIGINAL_PATH = str(Path(DATASET_FOLDER) / "genres_original")
IMAGES_ORIGINAL_PATH = str(Path(DATASET_FOLDER) / "images_original")
GENRE_LABELS = (
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock"
)


def _read_csv(filename: str) -> List[Dict[str, str]]:
    # assumes the first row is the field names

    full_path = str(Path(DATASET_FOLDER) / filename)

    rows = []
    with open(full_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            rows.append(row)

    return rows


def features_three_seconds_as_dict() -> List[Dict[str, str]]:
    """
    Load the `features_3_sec.csv` file as a list of dictionary objects,
    where each dictionary corresponds to a row mapping a field name to the
    value of the field in that row.
    """

    return _read_csv(filename=FEATURES_3_SEC_CSV)


def features_thirty_seconds_as_dict() -> List[Dict[str, str]]:
    """
    Load the `features_30_sec.csv` file as a list of dictionary objects,
    where each dictionary corresponds to a row mapping a field name to the
    value of the field in that row.
    """

    return _read_csv(filename=FEATURES_30_SEC_CSV)


def _get_filepaths_in_folder(folder_name: str, genre: str, extension: str) -> List[str]:

    full_path_to_folder = Path(folder_name) / genre

    files = [f for f in full_path_to_folder.iterdir() if f.is_file() and f.suffix == extension]

    return files


def audio_filepaths_as_dict() -> Dict[str, List[str]]:
    """
    Load a dictionary mapping genre to the list of paths to audio files in that
    genre. Uses the `genre_original` folder in the GTZAN dataset.
    """

    return {
        genre: _get_filepaths_in_folder(folder_name=GENRES_ORIGINAL_PATH, genre=genre, extension=".wav")
        for genre in GENRE_LABELS
    }


def image_filepaths_as_dict() -> Dict[str, List[str]]:
    """
    Load a dictionary mapping genre to the list of paths to spectogram image files in that genre.
    Uses the `image_original` folder in the GTZAN dataset.
    """

    return {
        genre: _get_filepaths_in_folder(folder_name=IMAGES_ORIGINAL_PATH, genre=genre, extension=".wav")
        for genre in GENRE_LABELS
    }
