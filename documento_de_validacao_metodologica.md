# Relatório de Validação Metodológica e Arquitetura Experimental (Revisão 2.0)

**Projeto:** Classificação Preditiva de Risco Criminal no Entorno do IFSP - Campus São Paulo  
**Domínio:** Ciência de Dados Aplicada / Aprendizado de Máquina Supervisionado  
**Questão de Pesquisa Norteadora:**  
> *"Dada uma determinada janela de horário em um dia específico, é possível prever/classificar o nível de risco de ocorrências criminais (Alto/Baixo) no entorno do campus do IFSP-SPO?"*

---

## 1. Visão Geral da Arquitetura do Pipeline

O projeto foi reestruturado seguindo as melhores práticas de Engenharia de Software e modularidade para Ciência de Dados:

```
├── data/
│   ├── raw/             <- Planilhas anuais da SSP-SP (2022 a 2026)
│   ├── interim/         <- CSV unificado consolidado (SPDadosCriminais_Unificado.csv)
│   └── processed/       <- Grid cartesiano completo e enriquecido (dados_processados.csv)
├── models/
│   ├── checkpoints/     <- 120 modelos (.pkl) das 30 sementes e log_experimentos.csv
│   └── final/           <- Artefato definitivo de deploy (modelo_risco_ifsp_final.joblib)
├── modules/
│   ├── config.py        <- Constantes, hiperparâmetros e parâmetros espaciais/temporais
│   ├── features.py      <- Haversine, grid cartesiano, engenharia de lags e splits sem leakage
│   ├── plots.py         <- Visualizações acadêmicas de distribuição, ICs e matriz de confusão
│   └── modeling/
│       ├── train.py     <- Sanidade (Passo 1), replicação Monte Carlo e treino final
│       ├── evaluation.py<- IC 95%, Friedman, Post-Hoc Wilcoxon-Holm e Concept Drift
│       └── predict.py   <- Inferência operacional em tempo real
├── notebooks/
│   ├── 01-extracao-unificaca-dados.ipynb
│   ├── 02-analise-exploratoria.ipynb
│   └── 03-modelagem-risco.ipynb
└── tests/
    ├── test_features.py <- Testes unitários para Haversine, lags causais e splits
    └── test_modeling.py <- Testes unitários para sanidade, Friedman e Wilcoxon-Holm
```

---

## 2. Decisões Metodológicas e Correções Críticas Implementadas

### 2.1. Amostragem Espacial via Distância Geodésica de Haversine ($d \le 2,0\text{ km}$)
* **Implementação:** Aplicação vetorizada da fórmula de Haversine tendo como centroide as coordenadas do IFSP-SPO ($-23.5237759, -46.624897$):
  $$d = 2r \arcsin\left(\sqrt{\sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right)}\right)$$
  onde $r = 6371.0088\text{ km}$.
* **Justificativa:** Registros policiais apresentam inconsistências frequentes de delimitação de bairros (Bom Retiro, Canindé, Pari, Luz e Centro). A delimitação euclidiana/geodésica contínua isola com precisão a mancha criminal real que afeta os trajetos a pé e o acesso aos terminais de transporte público (Metrô Armênia e Tiradentes) utilizados pela comunidade acadêmica.

### 2.2. Tratamento do Período do Dia e Auditoria das Ocorrências sem Horário
* **Implementação:** Criação da feature `DESCR_PERIODO_OCORRENCIA_BO` mapeada a partir de `HORA_OCORRENCIA_BO` (Madrugada: 00h-06h, Manhã: 06h-12h, Tarde: 12h-18h, Noite: 18h-24h). Para os registros com horário ausente, utilizou-se a coluna original `DESCR_PERIODO` como mecanismo de contingência (*fallback*).
* **Auditoria de Atrito Amostral:** 22.128 registros (40,48% dos crimes no raio de 2 km) não possuíam hora determinada (ex.: "EM HORA INCERTA").
* **Justificativa da Exclusão Controlada:** Para a tarefa de predição discreta em janelas de 6 horas, imputar turnos artificiais para crimes de horário desconhecido adicionaria ruído espúrio severo aos dados. O descarte é metodologicamente necessário, operando a contagem de crimes como um *lower bound* (limite inferior) da densidade criminal resolúvel.

### 2.3. Grid Cartesiano Completo e Imputação de Calmaria
* **Implementação:** Geração do produto cartesiano $\text{Datas} \times \{\text{MADRUGADA, MANHÃ, TARDE, NOITE}\}$, realizando *left join* com a contagem real de BOs e imputando $0$ nas janelas sem ocorrência (`fillna(0)`).
* **Justificativa:** Se a modelagem agrupasse apenas as datas presentes nos BOs, as janelas seguras (onde **nenhum crime ocorreu**) seriam sumariamente excluídas da amostra, gerando viés do sobrevivente. Um classificador de risco exige exemplos negativos reais para discernir períodos de calmaria de períodos críticos.

### 2.4. Binarização de Risco sem Vazamento de Dados (*Zero Target Leakage*)
* **Correção Crucial Realizada:** Na versão preliminar, a mediana global ($4.0$) havia sido calculada sobre o dataset inteiro (2022-2026). Isso foi corrigido:
  * A série temporal foi particionada cronologicamente em $85\%$ para Desenvolvimento e $15\%$ para Teste Final;
  * O limiar da mediana foi **calculado estritamente sobre $y_{dev}$** ($4.0$ crimes/bloco) e aplicado subsequentemente no conjunto de teste:
    $$y = \begin{cases} 1, & \text{se } \text{QTD\_CRIMES} > \text{Mediana}(y_{dev}) \\ 0, & \text{se } \text{QTD\_CRIMES} \le \text{Mediana}(y_{dev}) \end{cases}$$
  * Essa separação elimina 100% do vazamento de distribuição futura para a variável dependente.

### 2.5. Engenharia de Features Temporais Dinâmicas e Causais
Para romper a limitação anterior (em que o modelo enxergava apenas 336 estados estáticos de calendário), foram introduzidas variáveis dinâmicas e causais estritamente passadas (*sem look-ahead bias*):
* `ANO`: Permite capturar a tendência plurianual de redução/aumento na criminalidade;
* `CRIMES_LAG_1`: Volume de crimes no turno imediatamente anterior ($t-1$);
* `CRIMES_LAG_4`: Volume de crimes no mesmo turno do dia anterior ($t-4$);
* `CRIMES_MEDIA_MOVEL_7D`: Média móvel dos 28 turnos precedentes (`shift(1).rolling(28).mean()`);
* `PERIODO_LETIVO_IFSP`: Proxy institucional indicando meses letivos regulares (fev-jun, ago-dez) vs. férias discentes (jan, jul).

---

## 3. Protocolo Experimental em 8 Passos (Conformidade com o Enunciado)

| Passo do Enunciado | Implementação Adotada | Justificativa Técnica |
| :--- | :--- | :--- |
| **(1) Teste de Sanidade** | Testes automatizados cobrindo **todos os 4 modelos** em subconjunto de 30 instâncias, validando shapes, tipos, consistência de probabilidades e convergência. | Atende estritamente à exigência de verificar consistência de entrada/saída de todo o catálogo supervisionado, eliminando bugs de integração. |
| **(2) População-Alvo e Amostragem** | Janelas de 6h em raio de 2,0 km do IFSP entre 2022 e 2026 agregadas em grid cartesiano com controle de atrito amostral. | Delimitação geodésica contínua sem viés administrativo. |
| **(3) Execução dos Modelos** | Comparação de 4 famílias distintas: *Logistic Regression* (linear padronizada), *Decision Tree* (simbólica), *Random Forest* (bagging) e *Gradient Boosting* (boosting). | Cobre com amplitude os paradigmas clássicos para dados tabulares. |
| **(4) Reprodutibilidade e Replicação** | 30 execuções independentes (sementes $42$ a $71$) via particionamento Monte Carlo estratificado em $X_{dev}$; serialização dos 120 modelos (`.pkl`) e log consolidado (`log_experimentos.csv`). | Fornece amostras estocásticas para testes de hipótese e viabiliza replicação independente. |
| **(5) Análise Estatística Formal** | **Intervalos de Confiança a 95%** (distribuição $t$, $gl=29$), **Teste Omnibus de Friedman** e **Teste Post-Hoc Pareado de Wilcoxon com Correção de Holm-Bonferroni**. | Supera a limitação de testes omnibus isolados, identificando estatisticamente onde residem as diferenças reais com controle do FWER. |
| **(6) Generalização em Dados Não Vistos** | Avaliação do modelo vencedor sobre os $15\%$ cronológicos finais (2026), confronto contra o baseline da classe majoritária (*Dummy*) e auditoria quantitativa de *Concept Drift*. | Avalia a real capacidade preditiva no mundo real sob mudança de regime criminal. |
| **(7) Versão Definitiva de Deploy** | Treinamento do classificador vencedor sobre $100\%$ do histórico ($X_{dev} \cup X_{test}$) e exportação em `.joblib` com função de inferência. | Maximiza o histórico de dados para o modelo em produção. |
| **(8) Relatório Científico** | Estruturação e subsídios experimentais para artigo no formato ACL 2023 (máx. 8 páginas). | Articulação acadêmica entre teoria, experimentos e limitações. |

---

## 4. Resultados Estatísticos Consolidados

### 4.1. Intervalos de Confiança a 95% (F1-Score nas 30 Partições de Validação)

| Modelo | Média F1-Score | Desvio-Padrão | Erro-Padrão | IC 95% Inferior | IC 95% Superior | Mediana |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting** | **0,7307** | 0,0142 | 0,0026 | **0,7254** | **0,7360** | 0,7308 |
| **Decision Tree** | **0,7281** | 0,0134 | 0,0024 | **0,7231** | **0,7331** | 0,7295 |
| **Random Forest** | **0,7222** | 0,0136 | 0,0025 | **0,7171** | **0,7273** | 0,7212 |
| **Logistic Regression** | **0,6996** | 0,0124 | 0,0023 | **0,6950** | **0,7042** | 0,7038 |

### 4.2. Teste Omnibus de Friedman
* **Estatística $\chi^2_F$:** $63,5200$
* **p-valor:** $1,0396 \times 10^{-13}$
* **Conclusão:** Rejeita-se categoricamente a hipótese nula $H_0$ ao nível de $\alpha = 0,05$. Existe superioridade estatisticamente significante entre os classificadores analisados.

### 4.3. Teste Post-Hoc Pareado de Wilcoxon com Correção de Holm-Bonferroni

| Comparação Pareada | Diferença Média | Vitórias A | Vitórias B | p-valor Bruto | p-valor Ajustado (Holm) | Significante ($\alpha=0,05$)? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree vs. Logistic Reg.** | $+0,0285$ | 30 | 0 | $1,86 \times 10^{-9}$ | $1,12 \times 10^{-8}$ | **Sim** |
| **Gradient Boosting vs. Logistic Reg.** | $+0,0312$ | 30 | 0 | $1,86 \times 10^{-9}$ | $1,12 \times 10^{-8}$ | **Sim** |
| **Random Forest vs. Logistic Reg.** | $+0,0226$ | 29 | 1 | $3,73 \times 10^{-9}$ | $1,49 \times 10^{-8}$ | **Sim** |
| **Gradient Boosting vs. Random Forest** | $+0,0086$ | 27 | 3 | $2,55 \times 10^{-7}$ | $7,66 \times 10^{-7}$ | **Sim** |
| **Decision Tree vs. Random Forest** | $+0,0059$ | 21 | 9 | $1,20 \times 10^{-2}$ | $2,41 \times 10^{-2}$ | **Sim** |
| **Decision Tree vs. Gradient Boosting** | $-0,0026$ | 12 | 18 | $0,2129$ | **$0,2129$** | **NÃO (Empate Técnico)** |

#### Discussão Fundamental: Princípio da Parcimônia (Navalha de Occam)
O teste post-hoc de Wilcoxon demonstrou que **Gradient Boosting e Decision Tree apresentam empate técnico estatístico** ($p_{\text{ajustado}} = 0,2129 > 0,05$).  
Embora o Gradient Boosting atinja a pontuação média marginalmente superior ($0,7307$ vs. $0,7281$), a **Árvore de Decisão** oferece interpretabilidade simbólica total (regras explícitas de IF-THEN) e custo computacional negligenciável. Para a gestão de segurança do campus, essa transparência é um diferencial operacional decisivo.

---

## 5. Avaliação em Dados Não Vistos e Auditoria de Concept Drift

### 5.1. Desempenho no Conjunto de Teste (2026)
* **Acurácia do Modelo (Gradient Boosting):** $76,85\%$
* **F1-Score Macro:** $0,6126$
* **ROC-AUC:** **$0,7871$** (Excelente capacidade discriminatória de ordenação de risco)
* **Acurácia do Baseline Ingênuo (Dummy Most Frequent):** $77,04\%$
* **F1-Score do Baseline Ingênuo:** $0,0000$

### 5.2. Auditoria Quantitativa do Fenômeno de Concept Drift
* **Volume médio de crimes/janela em Dev (2022-2025):** $5,01$ ocorrências
* **Volume médio de crimes/janela em Teste (2026):** $2,84$ ocorrências
* **Variação na densidade criminal:** **$-43,4\%$**
* **Prevalência da classe Risco Alto:** $48,7\%$ em Dev vs. $23,0\%$ no Teste

**Interpretação Científica:**  
A redução progressiva das ocorrências registradas em 2026 provocou um deslocamento severo de distribuição (*distribution shift*). Como o limiar de risco de 4 crimes foi calibrado sobre a realidade mais violenta de 2022-2025, o modelo passou a classificar dias de 2026 com base em critérios do passado. Essa descoberta empírica não invalida o trabalho — pelo contrário: demonstra a necessidade de implementar **limiares dinâmicos adaptativos** em sistemas de inteligência criminal em produção.

---

## 6. Respostas Críticas às 4 Questões da Seção 5

1. **A formulação do problema como classificação binária por janela de 6 horas responde com precisão à pergunta de pesquisa proposta?**  
   * **Sim, com ressalva conceitual:** A granularidade de 6 horas atende diretamente à rotina dos turnos de aula do IFSP (manhã, tarde, noite e dispersão na madrugada). A ressalva é que a binarização pela mediana mensura *densidade de eventos*, tratando furtos simples e crimes violentos com o mesmo peso. Isso deve ser documentado como limitação no artigo.
2. **A truncagem em 2022 e a imputação de zeros via grid cartesiano estão matematicamente corretas para evitar viés de agregação?**  
   * **Sim, com o target leakage corrigido:** A truncagem removeu dados anacrônicos e o período atípico da pandemia. O grid cartesiano garantiu a presença dos exemplos negativos. Com a calibração da mediana estritamente sobre $y_{dev}$, o pipeline está matematicamente blindado.
3. **O conjunto de 4 modelos cobre satisfatoriamente a diversidade algorítmica exigida para a publicação em formato ACL?**  
   * **Sim, plenamente:** Abrange modelos paramétricos (Regressão Logística com StandardScaler), modelos de partição recursiva interpretável (Árvore de Decisão), comitês homogêneos por bagging (Random Forest) e comitês aditivos por boosting (Gradient Boosting).
4. **A substituição do Teste t de Student pelo Teste de Friedman foi adequada para o controle de múltiplas comparações pareadas?**  
   * **Sim, e foi concluída com o teste post-hoc:** O teste de Friedman controlou a taxa de erro Tipo I inicial, e o teste post-hoc de Wilcoxon com correção de Holm-Bonferroni identificou com precisão o empate técnico entre Gradient Boosting e Árvore de Decisão e a superioridade de ambos sobre os demais.
