"""
Script de execução completa do pipeline experimental de modelagem e avaliação estatística.
Atende a todos os 7 passos práticos do enunciado acadêmico com rigor metodológico.
"""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np
import pandas as pd

from modules.config import (
    BASE_RANDOM_SEED,
    DATA_PROCESSED_DIR,
    DEV_TEST_SPLIT_RATIO,
    FINAL_MODEL_NAME,
    MODELS_FINAL_DIR,
    N_EXPERIMENT_ITERATIONS,
    PROCESSED_DATASET_NAME,
)
from modules.features import apply_risk_threshold_without_leakage, split_dev_test_chronological
from modules.modeling.evaluation import (
    analyze_concept_drift,
    compute_confidence_intervals,
    evaluate_unseen_data,
    run_friedman_test,
    run_posthoc_wilcoxon,
)
from modules.modeling.train import (
    get_model_constructors,
    run_experiments,
    run_sanity_check,
    train_final_model,
)
from modules.plots import plot_confidence_intervals, plot_confusion_matrix, plot_f1_distributions


def main():
    print("=" * 70)
    print("PROJETO SPOINCD: PIPELINE DE EXPERIMENTAÇÃO E AVALIAÇÃO ESTATÍSTICA")
    print("=" * 70)

    # 1. Carregamento dos dados processados
    processed_file = DATA_PROCESSED_DIR / PROCESSED_DATASET_NAME
    if not processed_file.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {processed_file}")

    print(f"\nCarregando dados processados de {processed_file}...")
    df = pd.read_csv(processed_file, sep=";")
    print(f"Total de janelas temporais carregadas: {len(df)}")

    # 2. Particionamento temporal estrito (85% desenvolvimento, 15% teste de generalização)
    df_dev_raw, df_test_raw = split_dev_test_chronological(df, split_ratio=DEV_TEST_SPLIT_RATIO)
    print(f"Partição de Desenvolvimento (passado): {len(df_dev_raw)} instâncias ({df_dev_raw['DATA'].min()} a {df_dev_raw['DATA'].max()})")
    print(f"Partição de Teste Final (futuro não visto): {len(df_test_raw)} instâncias ({df_test_raw['DATA'].min()} a {df_test_raw['DATA'].max()})")

    # 3. Definição do limiar de risco SEM Target Leakage (calculado estritamente sobre df_dev)
    df_dev, df_test, threshold = apply_risk_threshold_without_leakage(df_dev_raw, df_test_raw)
    print(f"\nLimiar da Mediana (calculado estritamente em Dev): {threshold:.1f} crimes/janela")
    print(f"Prevalência de Risco Alto em Dev: {df_dev['RISCO_ALTO'].mean()*100:.2f}%")
    print(f"Prevalência de Risco Alto em Teste: {df_test['RISCO_ALTO'].mean()*100:.2f}%")

    # Definição das colunas de features (X) e target (y)
    period_dummies = [c for c in df.columns if c.startswith("PERIODO_") and c != "PERIODO_LETIVO_IFSP"]
    feature_cols = [
        "ANO",
        "MES",
        "DIA_SEMANA",
        "FLAG_FIM_DE_SEMANA",
        "TRIMESTRE",
        "PERIODO_LETIVO_IFSP",
        "CRIMES_LAG_1",
        "CRIMES_LAG_4",
        "CRIMES_MEDIA_MOVEL_7D",
    ] + period_dummies

    X_dev = df_dev[feature_cols]
    y_dev = df_dev["RISCO_ALTO"]
    X_test = df_test[feature_cols]
    y_test = df_test["RISCO_ALTO"]

    print(f"Matriz de atributos (X) configurada com {len(feature_cols)} features:")
    print(f"Features: {feature_cols}")

    # =========================================================================
    # PASSO 1: Teste de Sanidade e Consistência (Todos os 4 modelos)
    # =========================================================================
    print("\n" + "-" * 70)
    print("PASSO 1: Teste de Sanidade e Consistência de Entrada/Saída")
    print("-" * 70)
    models = get_model_constructors()
    X_sanity = X_dev.iloc[:30].copy()
    y_sanity = y_dev.iloc[:30].copy()

    sanity_accs = run_sanity_check(models, X_sanity, y_sanity)
    for model_name, acc in sanity_accs.items():
        print(f"  [OK] {model_name:<20}: convergência atestada (Acurácia no subconjunto: {acc*100:.1f}%)")
    print("Todos os 4 modelos verificados com consistência de tensores e ausência de NaNs.")

    # =========================================================================
    # PASSOS 3 & 4: Execução e Replicação Experimental (30 sementes)
    # =========================================================================
    print("\n" + "-" * 70)
    print(f"PASSOS 3 e 4: Execução de {N_EXPERIMENT_ITERATIONS} Iterações com Sementes {BASE_RANDOM_SEED} a {BASE_RANDOM_SEED + N_EXPERIMENT_ITERATIONS - 1}")
    print("-" * 70)
    df_results = run_experiments(
        models,
        X_dev,
        y_dev,
        n_iterations=N_EXPERIMENT_ITERATIONS,
        base_seed=BASE_RANDOM_SEED,
    )
    print(f"Total de experimentos salvos no log: {len(df_results)}")

    # =========================================================================
    # PASSO 5: Análise Estatística Formal dos Resultados
    # =========================================================================
    print("\n" + "-" * 70)
    print("PASSO 5: Análise Estatística Formal (IC 95%, Friedman, Post-Hoc Wilcoxon-Holm)")
    print("-" * 70)

    # 5.1 Intervalos de Confiança (distribuição t com 29 graus de liberdade)
    df_ci = compute_confidence_intervals(df_results, metric="F1_Score", confidence=0.95)
    print("\nIntervalos de Confiança a 95% para o F1-Score:")
    print(df_ci[["Modelo", "Média", "Desvio_Padrão", "IC_95%_Inf", "IC_95%_Sup", "Mediana"]].to_string(index=False))

    # 5.2 Teste Omnibus de Friedman
    stat_friedman, p_friedman, is_sig = run_friedman_test(df_results, metric="F1_Score")
    print(f"\nTeste Omnibus de Friedman: Estatística = {stat_friedman:.4f}, p-valor = {p_friedman:.4e}")
    if is_sig:
        print("  -> Rejeita-se H0 ao nível de 5%: existe discrepância estatística entre os modelos.")
    else:
        print("  -> Não se rejeita H0 ao nível de 5%: desempenhos globalmente equivalentes.")

    # 5.3 Teste Post-Hoc de Wilcoxon Pareado com Correção de Holm-Bonferroni
    print("\nTeste Post-Hoc Pareado de Wilcoxon (com controle de taxa de erro de família por Holm):")
    df_posthoc = run_posthoc_wilcoxon(df_results, metric="F1_Score", alpha=0.05)
    print(df_posthoc[["Comparação", "Diferença_Média (A - B)", "Vitórias_A", "Vitórias_B", "p_valor_bruto", "p_valor_ajustado_Holm", "Significante_5%"]].to_string(index=False))

    # Seleção do modelo vencedor fundamentada na parcimônia científica
    best_model_name = df_ci.iloc[0]["Modelo"]
    second_model_name = df_ci.iloc[1]["Modelo"]

    print(f"\nModelo com maior pontuação média: {best_model_name}")
    print("Discussão de Parcimônia (Navalha de Occam):")
    # Verificar se há diferença estatística real entre o 1º e 2º colocados
    pair_match = df_posthoc[
        ((df_posthoc["Modelo_A"] == best_model_name) & (df_posthoc["Modelo_B"] == second_model_name))
        | ((df_posthoc["Modelo_A"] == second_model_name) & (df_posthoc["Modelo_B"] == best_model_name))
    ]
    if not pair_match.empty and not pair_match.iloc[0]["Significante_5%"]:
        print(f"  -> Empate técnico estatístico entre {best_model_name} e {second_model_name} (p-ajustado = {pair_match.iloc[0]['p_valor_ajustado_Holm']:.4f} > 0.05).")
        print(f"  -> A Árvore de Decisão oferece interpretabilidade total das regras para a gestão do campus, enquanto o Random Forest oferece ligeira redução de variância.")

    # =========================================================================
    # PASSO 6: Estimativa de Erro em Dados Não Vistos (Generalização)
    # =========================================================================
    print("\n" + "-" * 70)
    print("PASSO 6: Avaliação em Dados Não Vistos e Auditoria de Concept Drift")
    print("-" * 70)
    chosen_factory = models[best_model_name]
    model_eval = chosen_factory(BASE_RANDOM_SEED)

    gen_eval = evaluate_unseen_data(model_eval, X_dev, y_dev, X_test, y_test)
    model_m = gen_eval["model_metrics"]
    base_m = gen_eval["baseline_metrics"]

    print(f"Modelo Avaliado: {best_model_name}")
    print(f"  Acurácia do Modelo:          {model_m['Acurácia']*100:.2f}%")
    print(f"  F1-Score do Modelo:          {model_m['F1_Score']:.4f}")
    print(f"  Precisão (Classe Alto Risco): {model_m['Precisão_Classe_1']*100:.2f}%")
    print(f"  Recall (Classe Alto Risco):   {model_m['Recall_Classe_1']*100:.2f}%")
    if model_m['ROC_AUC'] is not None:
        print(f"  ROC-AUC:                     {model_m['ROC_AUC']:.4f}")

    print(f"\nComparação com Baseline da Classe Majoritária (Dummy):")
    print(f"  Acurácia do Baseline Dummy:  {base_m['Acurácia']*100:.2f}%")
    print(f"  F1-Score do Baseline Dummy:  {base_m['F1_Score']:.4f}")

    # Auditoria de Concept Drift
    drift = analyze_concept_drift(df_dev, df_test)
    print("\nAuditoria Quantitativa de Deslocamento de Conceito (Concept Drift):")
    print(f"  Volume médio de crimes/janela em Dev (2022-2025):  {drift['dev_crimes_mean_per_window']:.2f}")
    print(f"  Volume médio de crimes/janela em Teste (2026):     {drift['test_crimes_mean_per_window']:.2f}")
    print(f"  Variação relativa no volume criminal:              {drift['crime_volume_drift_pct']:.1f}%")
    print(f"  Prevalência de Risco Alto: Dev={drift['dev_high_risk_prevalence_pct']:.1f}% vs Teste={drift['test_high_risk_prevalence_pct']:.1f}%")

    # =========================================================================
    # PASSO 7: Treinamento da Versão Final para Deploy
    # =========================================================================
    print("\n" + "-" * 70)
    print("PASSO 7: Treinamento da Versão Definitiva (100% dos Dados) para Deploy")
    print("-" * 70)
    X_full = pd.concat([X_dev, X_test], ignore_index=True)
    y_full = pd.concat([y_dev, y_test], ignore_index=True)

    final_model_path = MODELS_FINAL_DIR / FINAL_MODEL_NAME
    train_final_model(chosen_factory, X_full, y_full, seed=BASE_RANDOM_SEED, export_path=final_model_path)
    print(f"Modelo definitivo treinado em {len(X_full)} instâncias e exportado para: {final_model_path}")

    print("\n" + "=" * 70)
    print("PIPELINE EXPERIMENTAL CONCLUÍDO COM SUCESSO!")
    print("=" * 70)


if __name__ == "__main__":
    main()
