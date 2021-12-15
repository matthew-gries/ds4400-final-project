import sys
sys.path.append("../")

from typing import Dict, Tuple

import numpy as np
from ds4400_final_project.dataset.load_gtzan import load_data_from_file
from sklearn.model_selection import train_test_split



def train_and_evaluate_classifier(
    csv_filename: str,
    classifier,
    test_size: float = 0.33,
    random_state: int = 42,
) -> Tuple[float, float, Dict[str, int]]:

    # import the data from the seconds features CSV
    X, y, index_genre_map, genre_index_map = load_data_from_file(csv_filename)

    # split all the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    # train the classifier
    classifier.fit(X_train, y_train)

    # evaluate the model on training data
    train_pred = classifier.predict(X_train)
    train_failed = np.where(np.not_equal(y_train.ravel(), train_pred))
    y_train_failed = y_train[train_failed]

    # compute the accuracy / error on the training data
    train_accuracy = classifier.score(X_train, y_train)

    # evaluate the model on testing data
    test_pred = classifier.predict(X_test)
    test_failed = np.where(np.not_equal(y_test.ravel(), test_pred))
    y_test_failed = y_test[test_failed]

    # compute the accuracy / error on the testing data
    test_accuracy = classifier.score(X_test, y_test)

    failed_count: Dict[str, int] = {
        genre: 0 for genre in genre_index_map.keys()}

    for y_failed in [*y_train_failed, *y_test_failed]:
        genre = index_genre_map[y_failed]
        failed_count[genre] += 1

    return train_accuracy, test_accuracy, failed_count


def get_best_test_result(results: Dict[str, Dict[str, Tuple]]) -> Tuple:
	best_result = None
	for filename, val in results.items():
		for param, result in val.items():
			# save the first item by default
			if not best_result:
				best_result = filename, param, result
				continue

			_, param, (_, saved_test_acc, _) = best_result
			_, curr_test_acc, _ = result

			# if the current is better than the saved, update the saved
			if curr_test_acc > saved_test_acc:
				best_result = filename, param, result
	return best_result


def get_title_from_filename(filename: str) -> str:
    name, extension = filename.split(".")
    return name.replace("_", " ").title()


def print_classifier_results(
    title: str,
    train_accuracy: float,
    test_accuracy: float,
    failed_count: Dict[str, int]
) -> None:
    total_incorrect = sum(failed_count.values())

    print("=="*30)
    print(get_title_from_filename(title))
    print("=="*30)
    print(f"Train accuracy: {round(train_accuracy * 100, 2)}%")
    print(f"Train error:    {round((1-train_accuracy) * 100, 2)}%")
    print(f"Test accuracy:  {round(test_accuracy * 100, 2)}%")
    print(f"Test error:     {round((1-test_accuracy) * 100, 2)}%")
    print()
    print("| #   | Genre".ljust(18), "| # of Incorrect | % of All Incorrect |")
    print("| --- |    ---     |      ---       |        ---         |")

    failed = sorted([(genre, count) for genre, count in failed_count.items(
    )], key=lambda t: t[1], reverse=True)

    for i, (genre, failed_count) in enumerate(failed):
        print(f"| {i+1:02}  |", genre.upper().ljust(10), "|", str(failed_count).ljust(14),
              f"| {round((failed_count / total_incorrect) * 100, 1)}%".ljust(20), "|")
