"""
Script para enriquecer e atualizar o notebook 02-analise-exploratoria.ipynb com:
1. Auditoria formal dos 40,5% de crimes com período nulo/incerto;
2. Correção da tabela de contingência Período x Rúbrica;
3. Integração com as funções modulares de modules.features para exportação consistente.
"""

import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
NB_PATH = ROOT_DIR / "notebooks" / "02-analise-exploratoria.ipynb"

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total de células originais em 02: {len(nb['cells'])}")

# Localiza e atualiza a célula da tabela de contingência
for cell in nb["cells"]:
    if cell["cell_type"] == "code" and "pd.crosstab" in "".join(cell.get("source", "")):
        cell["source"] = [
            "# Tabela de contingência Período x Rúbrica utilizando a feature consolidada (com contingência/fallback)\n",
            "pd.crosstab(\n",
            "    index=df_crimes_filtrado[\"DESCR_PERIODO_OCORRENCIA_BO\"].fillna(\"HORA INCERTA / NÃO INFORMADO\"),\n",
            "    columns=df_crimes_filtrado[\"RUBRICA\"],\n",
            "    margins=True,\n",
            "    margins_name=\"Total\",\n",
            ")\n"
        ]
        print("Célula de crosstab atualizada com sucesso!")

# Localiza e atualiza a célula final de exportação
for cell in nb["cells"]:
    if cell["cell_type"] == "code" and "df_agrupado_final.to_csv" in "".join(cell.get("source", "")):
        cell["source"] = [
            "from modules.features import build_cartesian_grid, aggregate_crime_counts, engineer_features\n",
            "\n",
            "# Grid completo de datas e turnos\n",
            "grid_completo = build_cartesian_grid(\n",
            "    start_date=df_crimes_filtrado[\"DATA_OCORRENCIA_BO\"].dt.date.min(),\n",
            "    end_date=df_crimes_filtrado[\"DATA_OCORRENCIA_BO\"].dt.date.max(),\n",
            ")\n",
            "\n",
            "# Agregação com preenchimento de zeros para janelas seguras\n",
            "df_agrupado = aggregate_crime_counts(df_crimes_filtrado, grid_completo)\n",
            "\n",
            "# Engenharia de variáveis temporais dinâmicas e causais (sem target leakage)\n",
            "df_agrupado_final = engineer_features(df_agrupado)\n",
            "\n",
            "# Exportação com índice desativado\n",
            "arquivo_destino = pathlib.Path(DIR_DADOS_PROCESSADOS) / NOME_ARQUIVO_DATASET_PROCESSADO\n",
            "df_agrupado_final.to_csv(arquivo_destino, sep=\";\", index=False, encoding=\"utf-8\")\n",
            "print(f\"Dataset processado exportado com sucesso: {len(df_agrupado_final)} registros e {len(df_agrupado_final.columns)} colunas.\")\n",
            "df_agrupado_final.head()\n"
        ]
        print("Célula final de exportação atualizada com sucesso!")

# Inserção da seção de Auditoria de Dados Faltantes logo antes da bivariada
audit_md_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### Auditoria Metodológica: Ocorrências com Horário Ausente ou Incerto\n",
        "\n",
        "A análise de qualidade dos dados revelou que **22.128 ocorrências (40,48%)** no raio de 2,0 km do campus não possuem hora específica ou descrição textual de período (ex.: registros cadastrados como *\"EM HORA INCERTA\"*).\n",
        "\n",
        "**Decisão Metodológica e Justificativa:**\n",
        "1. Para a modelagem de risco preditivo discretizado por **janela de 6 horas**, a exclusão desses registros é uma necessidade técnica, pois alocar arbitrariamente um furto de hora desconhecida em um turno específico (ex.: imputar madrugada ou tarde) introduziria ruído artificial e distorceria os padrões sazonais reais.\n",
        "2. Como consequência empírica, a contagem de crimes por janela reflete os eventos temporalmente resolúveis, atuando como um *limite inferior (lower bound)* da criminalidade real.\n",
        "3. No artigo final (Seção de Limitações), esse viés de subnotificação temporal dos boletins da SSP-SP é formalmente documentado."
    ]
}

audit_code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Auditoria quantitativa das ocorrências sem período\n",
        "qtd_com_periodo = df_crimes_filtrado[\"DESCR_PERIODO_OCORRENCIA_BO\"].notna().sum()\n",
        "qtd_sem_periodo = df_crimes_filtrado[\"DESCR_PERIODO_OCORRENCIA_BO\"].isna().sum()\n",
        "total_filtrado = len(df_crimes_filtrado)\n",
        "\n",
        "print(f\"Total de crimes no raio de 2,0 km:        {total_filtrado}\")\n",
        "print(f\"Crimes com turno identificado:           {qtd_com_periodo} ({qtd_com_periodo/total_filtrado*100:.2f}%)\")\n",
        "print(f\"Crimes com hora incerta/nula (descarte): {qtd_sem_periodo} ({qtd_sem_periodo/total_filtrado*100:.2f}%)\")\n",
        "\n",
        "# Comparação das principais rúbricas entre os crimes com e sem período\n",
        "df_temp = df_crimes_filtrado.copy()\n",
        "df_temp[\"STATUS_HORA\"] = np.where(df_temp[\"DESCR_PERIODO_OCORRENCIA_BO\"].notna(), \"Com Horário\", \"Hora Incerta/Nula\")\n",
        "pd.crosstab(df_temp[\"RUBRICA\"], df_temp[\"STATUS_HORA\"], normalize=\"columns\").round(4) * 100\n"
    ]
}

# Encontrar o índice da célula de Análise Bivariada para inserir a auditoria antes
insert_idx = None
for idx, cell in enumerate(nb["cells"]):
    if "Análises bivariadas" in "".join(cell.get("source", "")):
        insert_idx = idx
        break

if insert_idx is not None:
    nb["cells"].insert(insert_idx, audit_code_cell)
    nb["cells"].insert(insert_idx, audit_md_cell)
    print(f"Seção de auditoria inserida na posição {insert_idx}!")

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print(f"Notebook 02 atualizado com sucesso em {NB_PATH}!")
