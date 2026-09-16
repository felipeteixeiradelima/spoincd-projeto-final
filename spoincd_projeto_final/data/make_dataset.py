from pathlib import Path

import urllib3

from spoincd_projeto_final.core import DatasetFile
from spoincd_projeto_final.util import logging_utils

urllib3.disable_warnings()

logger = logging_utils.get_logger(__name__)

_INITIAL_DATASET_YEAR = "2022"
_FINAL_DATASET_YEAR = "-1"  # -1 for latest year available

_BASE_URL = "https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados"
_BASE_DATASET_NAME = "SPDadosCriminais_{}.xlsx"


_PROXIES = {
    "http": "http://proxy.spo.ifsp.edu.br:3128",
    "https": "http://proxy.spo.ifsp.edu.br:3128",
}

_BASE_DATASET_DESTINATION_PATH = Path("./data/raw")
_TIMEOUT = 120


def _get_dataset_file(year: int | str) -> DatasetFile | None:
    dataset_file = DatasetFile(
        filename=_BASE_DATASET_NAME.format(year),
        dest_dir_path=_BASE_DATASET_DESTINATION_PATH,
    )

    try:
        dataset_file.download_dataset(base_url=_BASE_URL, proxies=_PROXIES, timeout=_TIMEOUT)
        return dataset_file
    except Exception:
        logger.exception("Error downloading dataset from %s", year)
        return None


def _get_all_dataset_files() -> list[DatasetFile]:
    dataset_files: list[DatasetFile] = []

    initial_year = int(_INITIAL_DATASET_YEAR)
    final_year = int(_FINAL_DATASET_YEAR)
    latest_year_flag = str(final_year) == "-1"

    if initial_year > final_year and not latest_year_flag:
        raise ValueError("Initial dataset year must be less than or equal to final dataset year")

    year = initial_year

    while latest_year_flag or year <= final_year:
        logger.debug("Downloading dataset from year %s", year)

        dataset_file = _get_dataset_file(year)

        if dataset_file is None:
            logger.debug("Failed to download %s", year)
            break

        logger.debug("Dataset from year %s downloaded", year)

        dataset_files.append(dataset_file)

        year += 1

    return dataset_files


def _save_dataset_files(dataset_files: list[DatasetFile]):
    for dataset_file in dataset_files:
        dataset_file.save_dataset()


def main():
    dataset_files = _get_all_dataset_files()
    _save_dataset_files(dataset_files)


if __name__ == "__main__":
    main()
