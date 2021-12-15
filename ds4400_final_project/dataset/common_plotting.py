import sys
sys.path.append("../")

from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from ds4400_final_project.dataset.common import get_title_from_filename



def plot_param_comparison_per_dataset(title: str, xlabel: str, results: Dict[str, Dict[str, Tuple]], show: bool = True):
    # { CSV_FILENAME: { SOLVER: ( TRAIN_ACC, TEST_ACC, FAILED_COUNT ) }

    for filename, result in results.items():
        name = get_title_from_filename(filename)
        x, y_train, y_test = [], [], []

        for param, (train_acc, test_acc, _) in result.items():
            x.append(param)
            y_train.append(train_acc * 100)
            y_test.append(test_acc * 100)

        y_pos = np.arange(len(x))

        plt.bar(y_pos, y_train, align="center", alpha=0.8,
                color="orange", label="Train Accuracy")
        plt.bar(y_pos, y_test, align="center", alpha=0.7,
                color="green", label="Test Accuracy")
        plt.xticks(y_pos, x)
        plt.ylim([0, 100])
        plt.ylabel("Accuracy (%)")
        plt.xlabel(xlabel)
        plt.title(f"{title} ({name})")
        plt.legend()

        if show:
            plt.show()
