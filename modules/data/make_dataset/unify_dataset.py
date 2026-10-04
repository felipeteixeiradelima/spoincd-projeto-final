import re
from pathlib import Path

import pandas as pd

from modules.util import logging_utils

logger = logging_utils.get_logger(__name__)


def find_xlsx_files(dir: Path) -> list[Path]:
    logger.info("Finding .xlsx files in '%s'...", dir.resolve())
    files = sorted(dir.glob("*.xlsx"))

    if not files:
        error_message = f"No .xlsx files found in '{dir}'."
        logger.error(error_message)
        raise FileNotFoundError(error_message)

    logger.info("Found %d file(s) to process.", len(files))
    return files


def extract_year_from_filename(filename: str) -> int | None:
    match = re.search(r"\d{4}", filename)
    if match:
        year = int(match.group())
        logger.debug("Identified year %d for file '%s'.", year, filename)
        return year

    logger.warning("Cound not indentify year in filename '%s'.", filename)
    return None


def read_semetral_sheets(filepath: Path) -> tuple[list[pd.DataFrame], int]:
    """
    Lê a 2ª e a 3ª abas de um arquivo Excel (semestres), adicionando metadados de origem.
    Retorna os DataFrames lidos e o total de linhas do arquivo.
    """
    logger.info("Processing file '%s'", filepath.name)
    ref_year = extract_year_from_filename(filepath.name)

    excel_obj = pd.ExcelFile(filepath, engine="openpyxl")
    sheets = excel_obj.sheet_names
    logger.debug("Sheets found in '%s': %s.", filepath.name, sheets)

    if len(sheets) < 2:
        logger.warning(
            "File '%s' countains only %d sheet(s); at least 3 were expected (description + 2 semesters).",
            filepath.name,
            len(sheets),
        )

    dfs_files = []
    total_file_rows = 0
    semester_sheets = sheets[1:]

    for semester_idx, sheet_name in enumerate(semester_sheets, start=1):
        logger.info("Reading sheet '%s' (Semester nº %d)...", sheet_name, semester_idx)
        df_sheet = pd.read_excel(excel_obj, sheet_name=sheet_name)

        # Cleaning header
        df_sheet.columns = df_sheet.columns.astype(str).str.strip()

        # Metadata
        df_sheet["ano_referencia"] = ref_year
        df_sheet["semestre_referencia"] = semester_idx
        df_sheet["arquivo_origem"] = filepath.name

        n_rows = len(df_sheet)
        total_file_rows += n_rows
        dfs_files.append(df_sheet)

        logger.debug(
            "Sheet '%s' loaded: %d rows and %d columns.",
            sheet_name,
            n_rows,
            len(df_sheet.columns),
        )

    return dfs_files, total_file_rows


def concat_and_validate(
    dfs: list[pd.DataFrame], num_expected_rows: int
) -> pd.DataFrame:
    logger.info("Contatenating %d blocks of data...", len(dfs))
    df_full = pd.concat(dfs, ignore_index=True)

    n_rows = len(df_full)
    if n_rows != num_expected_rows:
        error_message = (
            f"Integrity error: Expected {num_expected_rows} rows, "
            f"but full DataFrame has {n_rows}."
        )
        logger.error(error_message)
        raise AssertionError(error_message)

    logger.info("Integrity checked: %d rows in total.", n_rows)
    return df_full


def export_to_csv(
    df: pd.DataFrame,
    dest_path: Path,
    separator: str = ";",
    encoding: str = "utf-8",
) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Exporting consolidated data do CSV in '%s' (sep='%s', encoding='%s')...",
        dest_path.resolve(),
        separator,
        encoding,
    )

    df.to_csv(
        dest_path,
        sep=separator,
        index=False,
        encoding=encoding,
    )

    logger.info(
        "CSV files exported successfully: '%s' (%s linhas).",
        dest_path.name,
        f"{len(df):,}",
    )


def unify_dataset(src_dir_path: str, dest_file_path: str) -> None:
    src_dir = Path(src_dir_path)
    dest_file = Path(dest_file_path)

    files = find_xlsx_files(src_dir)

    dfs: list[pd.DataFrame] = []
    num_expected_rows = 0

    for arq in files:
        dfs_file, num_rows_file = read_semetral_sheets(arq)
        dfs.extend(dfs_file)
        num_expected_rows += num_rows_file

    df_full = concat_and_validate(dfs, num_expected_rows)
    export_to_csv(df_full, dest_file)
