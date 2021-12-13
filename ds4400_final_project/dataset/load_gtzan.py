from ds4400_final_project.dataset.constants import DATASET_FOLDER
from pathlib import Path
from typing import Tuple, Dict
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler

def load_data_from_file(csv_filename: str) -> Tuple[np.array, np.array, Dict[int, str], Dict[str, int]]:
	""" Load the CSV file from the dataset folder. """
	file = str(Path(DATASET_FOLDER) / csv_filename)
	features_list = np.genfromtxt(
		file, dtype=None, encoding=None, delimiter=",", skip_header=1, usecols=range(2, 60))
	features = np.array([list(x) for x in features_list])

	# Create a mapping between a numeric value and genre
	index_genre_map = {i: genre for i,
                    genre in enumerate(np.unique(features[:, -1]))}
	genre_index_map = {value: key for key, value in index_genre_map.items()}

	# split the inputs and their labels
	X = features[:, :57]
	y = np.array([genre_index_map[genre] for genre in features[:, -1]])

	# normalize the data
	X = normalize(X, axis=0)

	scaler = StandardScaler()
	X = scaler.fit_transform(X)

	return X, y, index_genre_map, genre_index_map