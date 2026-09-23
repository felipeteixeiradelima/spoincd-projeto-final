import argparse
import json
import os
from pathlib import Path

from requests.exceptions import HTTPError
import urllib3

from spoincd_projeto_final.core import DatasetFile
from spoincd_projeto_final.util import logging_utils

urllib3.disable_warnings()

logger = logging_utils.get_logger(__name__)

_INITIAL_DATASET_YEAR = 2022

_BASE_URL = "https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados"
_BASE_DATASET_NAME = "SPDadosCriminais_{}.xlsx"


def _get_dataset_file(
    year: int | str,
    dest_dir_path: str | os.PathLike[str] | Path,
    proxies: dict | None = None,
    timeout: float | tuple | None = None,
) -> DatasetFile | None:
    dataset_file = DatasetFile(
        filename=_BASE_DATASET_NAME.format(year),
        dest_dir_path=dest_dir_path,
    )

    try:
        dataset_file.download_dataset(base_url=_BASE_URL, proxies=proxies, timeout=timeout)
        return dataset_file
    except HTTPError as e:
        if e.response.status_code == 404:
            logger.warning("Dataset for year %s not found (HTTP 404)", year)
            return None

        logger.exception("Error downloading dataset from %s", year)
        return None


def _get_all_dataset_files(
    dest_dir_path: str | os.PathLike[str] | Path,
    initial_year: int | str = _INITIAL_DATASET_YEAR,
    final_year: int | str | None = None,
    proxies: dict | None = None,
    timeout: float | tuple | None = None,
) -> list[DatasetFile]:
    dataset_files: list[DatasetFile] = []

    initial_year = int(initial_year)

    latest_year_flag = final_year is None

    if not latest_year_flag and initial_year > final_year:
        raise ValueError("Initial dataset year must be less than or equal to final dataset year")

    if not latest_year_flag:
        final_year = int(final_year)

    year = initial_year

    while latest_year_flag or year <= final_year:
        logger.debug("Downloading dataset from year %s", year)

        dataset_file = _get_dataset_file(year, dest_dir_path, proxies, timeout)

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


def make_dataset(
    dest_dir_path: str | os.PathLike[str] | Path,
    proxies: dict | None = None,
    timeout: float | tuple | None = None,
):
    dataset_files = _get_all_dataset_files(dest_dir_path, proxies=proxies, timeout=timeout)
    _save_dataset_files(dataset_files)


def main():
    parser = argparse.ArgumentParser(description="Download the crime datasets.")
    parser.add_argument(
        "--dest-dir-path",
        default="./data/raw",
        help="Directory where downloaded datasets will be saved.",
    )
    parser.add_argument(
        "--proxies",
        type=json.loads,
        help='Proxy configuration as a JSON object, for example: {"https": "http://proxy:8080"}.',
    )
    parser.add_argument(
        "--timeout",
        type=float,
        help="Request timeout in seconds.",
    )

    args = parser.parse_args()
    make_dataset(
        dest_dir_path=args.dest_dir_path,
        proxies=args.proxies,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
