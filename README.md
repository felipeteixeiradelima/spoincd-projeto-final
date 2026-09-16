# spoincd-projeto-final

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Projeto final da disciplina Introdução à Ciência de Dados do curso Bacharelado em Sistemas de Informação do IFSP-SPO.

## Organização do Projeto

```
├── LICENSE            <- Licença open-source, caso tenha sido escolhida
├── Makefile           <- Arquivo com comandos práticos como `make data` ou `make train`
├── README.md          <- README principal para desenvolvedores que usam este projeto.
├── data
│   ├── external       <- Dados de fontes de terceiros.
│   ├── interim        <- Dados intermediários que foram transformados.
│   ├── processed      <- Conjuntos de dados finais e canônicos para modelagem.
│   └── raw            <- Dump original e imutável dos dados.
│
├── docs               <- Um projeto mkdocs padrão; consulte www.mkdocs.org para mais detalhes
│
├── models             <- Modelos treinados e serializados, previsões de modelos ou resumos de modelos
│
├── notebooks          <- Notebooks Jupyter. A convenção de nomenclatura usa um número para ordenação,
│                         as iniciais do criador e uma descrição curta separada por `-`, por exemplo
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Arquivo de configuração do projeto com metadados do pacote
│                         para spoincd_projeto_final e configuração de ferramentas como black
│
├── references         <- Dicionários de dados, manuais e todos os outros materiais explicativos.
│
├── reports            <- Análises geradas em HTML, PDF, LaTeX, etc.
│   └── figures        <- Gráficos e figuras gerados para uso em relatórios
│
└── spoincd_projeto_final   <- Código-fonte para uso neste projeto.
    │
    ├── __init__.py             <- Torna spoincd_projeto_final um módulo Python
    │
    ├── config.py               <- Armazena variáveis úteis e configurações
    │
    ├── dataset.py              <- Scripts para baixar ou gerar dados
    │
    ├── features.py             <- Código para criar recursos para modelagem
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Código para executar inferência de modelos com modelos treinados
    │   └── train.py            <- Código para treinar modelos
    │
    └── plots.py                <- Código para criar visualizações
```

--------

