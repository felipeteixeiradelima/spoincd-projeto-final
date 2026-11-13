# spoincd-projeto-final

Projeto final da disciplina **Introdução à Ciência de Dados** do curso Bacharelado em Sistemas de Informação do IFSP-SPO.

**Questão de Pesquisa:**
> *"Dada uma determinada janela de horário em um dia específico, é possível prever/classificar o nível de risco de ocorrências criminais (Alto/Baixo) no entorno do campus do IFSP-SPO?"*

---

## 👥 Autores

- **Andrey Gustavo Segantin dos Santos** - *SP3155439*  
  [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/AndreyGustavo267)
  [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/andrey-gustavo-s-dos-santos-6691a333a)

- **Elton Lima Araujo** - *SP3155544*  
  [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Elton1466)
  [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/elton-lima-araujo)

- **Felipe Teixeira de Lima** - *SP3155048*  
  [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/felipeteixeiradelima)
  [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/felipe-teixeira-de-lima)

- **João Gustavo dos Santos** - *SP3154211*  
  [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/joaogust)
  [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/joao-gustavo-dos-santos)

---

## 🗂️ Organização do Projeto

```
├── LICENSE                                    <- Licença open-source do projeto
├── README.md                                  <- Documentação geral e guia de execução
├── documento_de_validacao_metodologica.md     <- Relatório metodológico detalhado e validações
├── enunciado.md                               <- Requisitos e passos para execução do experimento
├── requirements.txt                           <- Dependências do projeto
│
├── data/
│   ├── raw/                                   <- Dados brutos baixados da SSP-SP (2022 a 2026)
│   ├── interim/                               <- Base unificada consolidada (SPDadosCriminais_Unificado.csv)
│   └── processed/                             <- Dataset final canônico (dados_processados.csv)
│
├── models/
│   ├── checkpoints/                           <- 120 modelos treinados (.pkl) das 30 sementes e log
│   └── final/                                 <- Artefato definitivo de deploy (modelo_risco_ifsp_final.joblib)
│
├── modules/                                   <- Pacote Python modular
│   ├── __init__.py                            <- Inicialização do pacote
│   ├── config.py                              <- Constantes de caminhos, coordenadas e hiperparâmetros
│   ├── features.py                            <- Distância de Haversine, grid cartesiano e lags dinâmicos
│   ├── plots.py                               <- Visualizações acadêmicas (distribuições, ICs e matrizes)
│   └── modeling/
│       ├── __init__.py                        <- Exportação centralizada das funções de modelagem
│       ├── train.py                           <- Testes de sanidade, replicação Monte Carlo e treino final
│       ├── evaluation.py                      <- IC 95%, Friedman, Post-Hoc Wilcoxon-Holm e Concept Drift
│       └── predict.py                         <- Rotina de inferência operacional e predição
│
├── notebooks/                                 <- Notebooks Jupyter sequenciais
│   ├── 01-extracao-unificaca-dados.ipynb      <- Download multithread e unificação das planilhas da SSP
│   ├── 02-analise-exploratoria.ipynb          <- Auditoria de qualidade, mapas de calor e grid cartesiano
│   └── 03-modelagem-risco.ipynb               <- Execução dos passos 1 a 7 do protocolo experimental
│
├── scripts/                                   <- Utilitários e pipelines de automação
│   ├── reprocess_data.py                      <- Regenera o dataset canônico com lags e sem target leakage
│   ├── run_experiments.py                     <- Executa o pipeline completo de modelagem e testes estatísticos
│   ├── update_notebook_02.py                  <- Atualizador do notebook 02
│   ├── update_notebook_03.py                  <- Atualizador do notebook 03
│   └── execute_notebook.py                    <- Executor programático de notebooks
│
└── tests/                                     <- Testes unitários automatizados
    ├── test_features.py                       <- Testes de Haversine, turnos e lags causais
    └── test_modeling.py                       <- Testes de sanidade, Friedman e Wilcoxon-Holm
```

---

## 🧪 Como Executar

### 1. Instalar Dependências
```bash
python -m venv .venv
# Ativar venv no Windows:
.\.venv\Scripts\activate
# No Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Executar a Suíte de Testes Unitários
```bash
python -m unittest discover tests
```

### 3. Executar o Pipeline Experimental Completo
```bash
python scripts/run_experiments.py
```

---

## 📊 Síntese dos Resultados Experimentais

* **Classificador Vencedor:** *Gradient Boosting* e *Decision Tree* (Empate Técnico Estatístico no teste Post-Hoc de Wilcoxon com $p_{\text{ajustado}} = 0,2129 > 0,05$).
* **F1-Score na Validação Cruzada (30 Partições):**
  * Gradient Boosting: $0,7307$ (IC 95%: $[0,7254;\; 0,7360]$)
  * Decision Tree: $0,7281$ (IC 95%: $[0,7231;\; 0,7331]$)
  * Random Forest: $0,7222$ (IC 95%: $[0,7171;\; 0,7273]$)
  * Logistic Regression: $0,6996$ (IC 95%: $[0,6950;\; 0,7042]$)
* **Teste Omnibus de Friedman:** Rejeição de $H_0$ ($p = 1,0396 \times 10^{-13}$).
* **Desempenho em Dados Não Vistos (2026):**
  * ROC-AUC: **$0,7871$**
  * Acurácia: $76,85\%$
  * Concept Drift detectado: queda de $-43,4\%$ no volume criminal médio no período pós-2025.
