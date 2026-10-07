import os

from . import download_dataset, unify_dataset


def make_dataset(
    raw_data_path: str | os.PathLike = "./data/raw",
    processed_data_path: str | os.PathLike = "./data/processed",
    dataset_filename: str = "SPDadosCriminais_Consolidado.csv",
    proxies: dict | None = None,
    timeout: float | tuple | None = None,
) -> None:
    """Downloads the dataset from the official source, unifies it, and saves it as a CSV file.

    Parameters
    ----------
    raw_data_path : str | os.PathLike, optional
        Directory where the downloaded spreadsheets will be saved, by default "./data/raw"
    processed_data_path : str | os.PathLike, optional
        Directory where the consolidated CSV will be created, by default "./data/processed"
    dataset_filename : str, optional
        Name of the output CSV file, by default "SPDadosCriminais_Consolidado.csv"
    proxies : dict | None, optional
        Proxies used in the download requests, by default None
    timeout : float | tuple | None, optional
        Timeout for the download requests, by default None
    """
    dest_file_path = os.path.join(processed_data_path, dataset_filename)

    download_dataset.download_dataset(raw_data_path, proxies, timeout)
    unify_dataset.unify_dataset(raw_data_path, dest_file_path)
