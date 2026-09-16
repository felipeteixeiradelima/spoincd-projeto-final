import os
from pathlib import Path

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from spoincd_projeto_final.util import logging_utils

logger = logging_utils.get_logger(__name__)


class DatasetFile:
    def __init__(
        self,
        filename: str,
        dest_dir_path: str | os.PathLike[str] | Path,
        content: bytes | None = None,
    ) -> None:
        self.filename = filename
        self.filepath = Path(dest_dir_path) / filename
        self.content = content

    def __repr__(self) -> str:
        return f"<DatasetFile(filename={self.filename}, filepath={self.filepath})>"

    def __str__(self):
        return f"<DatasetFile(filename={self.filename}, filepath={self.filepath})>"

    def __len__(self):
        return len(self.content) if self.content is not None else 0

    def is_downloaded(self) -> bool:
        return self.content is not None

    def is_saved(self) -> bool:
        return self.filepath.exists()

    @retry(stop=stop_after_attempt(3), reraise=True, wait=wait_exponential())
    def download_dataset(self, base_url: str, proxies: dict, timeout: float | tuple) -> None:
        if self.is_downloaded():
            logger.debug("%s is already downloaded", self)
            return

        if self.is_saved():
            logger.debug("%s is already saved", self)
            return

        url = f"{base_url}/{self.filename}"

        logger.debug("Downloading %s", self)

        response = requests.get(url, verify=False, proxies=proxies, timeout=timeout)

        logger.debug("%s downloaded", self)

        response.raise_for_status()

        self.content = response.content

    def save_dataset(self) -> None:
        if self.is_saved():
            logger.debug("%s is already saved", self)
            return

        logger.debug("Saving %s", self)

        self.filepath.write_bytes(self.content)

        logger.debug("%s saved", self)
