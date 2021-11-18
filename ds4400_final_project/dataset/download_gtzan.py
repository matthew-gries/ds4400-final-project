from kaggle.api.kaggle_api_extended import KaggleApi

from ds4400_final_project.dataset.constants import DATASET_SLUG, DOWNLOAD_FOLDER


def download_gtzan():
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(
        dataset=DATASET_SLUG,
        path=DOWNLOAD_FOLDER,
        force=False,
        quiet=False,
        unzip=True
    )


if __name__ == "__main__":
    download_gtzan()