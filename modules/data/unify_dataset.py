import re
from pathlib import Path

import pandas as pd

from modules.util import logging_utils

logger = logging_utils.get_logger(__name__)


def encontrar_arquivos_xlsx(diretorio: Path) -> list[Path]:
    """Localiza e ordena todos os arquivos .xlsx do diretório."""
    logger.info("Buscando arquivos .xlsx em: '%s'", diretorio.resolve())
    arquivos = sorted(diretorio.glob("*.xlsx"))

    if not arquivos:
        msg_erro = f"Nenhum arquivo .xlsx encontrado em: '{diretorio}'"
        logger.error(msg_erro)
        raise FileNotFoundError(msg_erro)

    logger.info("Encontrado(s) %d arquivo(s) para processamento.", len(arquivos))
    return arquivos


def extrair_ano_arquivo(nome_arquivo: str) -> int | None:
    """Extrai o ano (4 dígitos) do nome do arquivo."""
    match = re.search(r"\d{4}", nome_arquivo)
    if match:
        ano = int(match.group())
        logger.debug("Ano %d identificado para o arquivo '%s'", ano, nome_arquivo)
        return ano

    logger.warning("Ano não identificado no nome do arquivo: '%s'", nome_arquivo)
    return None


def ler_abas_semestrais(caminho_arquivo: Path) -> tuple[list[pd.DataFrame], int]:
    """
    Lê a 2ª e a 3ª abas de um arquivo Excel (semestres), adicionando metadados de origem.
    Retorna os DataFrames lidos e o total de linhas do arquivo.
    """
    logger.info("Processando arquivo: '%s'", caminho_arquivo.name)
    ano_referencia = extrair_ano_arquivo(caminho_arquivo.name)

    excel_obj = pd.ExcelFile(caminho_arquivo, engine="openpyxl")
    todas_abas = excel_obj.sheet_names
    logger.debug("Abas encontradas em '%s': %s", caminho_arquivo.name, todas_abas)

    if len(todas_abas) < 3:
        logger.warning(
            "O arquivo '%s' possui apenas %d aba(s). Esperava-se ao menos 3 (descritiva + 2 semestres).",
            caminho_arquivo.name,
            len(todas_abas),
        )

    dfs_arquivo = []
    total_linhas_arquivo = 0
    abas_semestres = todas_abas[1:3]

    for semestre_idx, nome_aba in enumerate(abas_semestres, start=1):
        logger.info("Lendo aba '%s' (Semestre %d)...", nome_aba, semestre_idx)
        df_aba = pd.read_excel(excel_obj, sheet_name=nome_aba)

        # Sanitiza cabeçalhos removendo espaços nas pontas
        df_aba.columns = df_aba.columns.astype(str).str.strip()

        # Metadados de rastreabilidade
        df_aba["ano_referencia"] = ano_referencia
        df_aba["semestre_referencia"] = semestre_idx
        df_aba["arquivo_origem"] = caminho_arquivo.name

        qtd_linhas = len(df_aba)
        total_linhas_arquivo += qtd_linhas
        dfs_arquivo.append(df_aba)

        logger.debug(
            "Aba '%s' carregada: %d linhas e %d colunas.",
            nome_aba,
            qtd_linhas,
            len(df_aba.columns),
        )

    return dfs_arquivo, total_linhas_arquivo


def concatenar_e_validar(dfs: list[pd.DataFrame], total_esperado: int) -> pd.DataFrame:
    """Concatena os DataFrames e valida a integridade da contagem de linhas."""
    logger.info("Concatenando %d blocos de dados...", len(dfs))
    df_consolidado = pd.concat(dfs, ignore_index=True)

    linhas_finais = len(df_consolidado)
    if linhas_finais != total_esperado:
        msg_erro = (
            f"Falha de integridade: Esperadas {total_esperado} linhas, "
            f"mas o DataFrame consolidado tem {linhas_finais}."
        )
        logger.error(msg_erro)
        raise AssertionError(msg_erro)

    logger.info("Integridade verificada com sucesso: %d linhas totais.", linhas_finais)
    return df_consolidado


def salvar_em_csv(
    df: pd.DataFrame,
    caminho_saida: Path,
    separador: str = ";",
    codificacao: str = "utf-8",
) -> None:
    """
    Exporta o DataFrame consolidado para .csv.
    Por padrão usa ponto e vírgula (;) e codificação utf-8 para preservar acentuações.
    """
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Exportando dados consolidados para CSV em: '%s' (sep='%s', encoding='%s')...",
        caminho_saida.resolve(),
        separador,
        codificacao,
    )

    df.to_csv(
        caminho_saida,
        sep=separador,
        index=False,
        encoding=codificacao,
    )

    logger.info(
        "Arquivo CSV exportado com sucesso: '%s' (%s linhas).",
        caminho_saida.name,
        f"{len(df):,}",
    )


def consolidar_excels(diretorio_origem: str, arquivo_saida: str) -> None:
    """Fluxo orquestrador da consolidação dos dados anuais para CSV."""
    logger.info("=== Iniciando Pipeline de Consolidação ===")
    diretorio = Path(diretorio_origem)
    destino = Path(arquivo_saida)

    arquivos = encontrar_arquivos_xlsx(diretorio)

    todos_dfs: list[pd.DataFrame] = []
    total_linhas_esperadas = 0

    for arq in arquivos:
        dfs_arquivo, linhas_arquivo = ler_abas_semestrais(arq)
        todos_dfs.extend(dfs_arquivo)
        total_linhas_esperadas += linhas_arquivo

    df_consolidado = concatenar_e_validar(todos_dfs, total_linhas_esperadas)
    salvar_em_csv(df_consolidado, destino)
    logger.info("=== Pipeline Concluído com Sucesso ===")


if __name__ == "__main__":
    consolidar_excels(
        diretorio_origem="./data/raw",
        arquivo_saida="./data/interim/dataset_consolidado.csv",
    )
