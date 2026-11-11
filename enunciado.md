**As atividades de avaliação devem ser entregues atravês da tarefa aberta aqui. Siga o fluxo abaixo para execução do experimento:**

**\# Passos para Execução de um Experimento**  
**\# (1) Testar a implementação do modelo**  
**\# Antes de iniciar a experimentação, verifique se o modelo está corretamente implementado. Isso inclui testes unitários, verificação da consistência de entrada e saída, e análise de overfitting em pequenos subconjuntos para validar o funcionamento.**  
**\# (2) Coletar uma amostra dos dados**  
**\# Defina a população-alvo e descreva o processo de amostragem.**   **Certifique-se de que a amostra é representativa do problema que está sendo estudado.**  
**\# (3) Executar o experimento**  
**\# Realize os testes de desempenho do modelo conforme o plano **experimental. Isso inclui treinar, validar e avaliar o modelo em múltiplas execuções.**  
**\# (4) Reproduzir o experimento (Reprodução/Replicação)**  
**\# Realize múltiplas execuções com diferentes sementes aleatórias e **particionamentos dos dados.**  
**\# Teste o modelo em múltiplos conjuntos de validação, gerando uma **amostra de erros.**  
**\# Salve os resultados intermediários: como as sementes dos geradores aleatórios, os pesos dos modelos treinados, métricas parciais e logs.**  
**\# Isso garante que o experimento pode ser replicado no futuro.**  
**\# (5) Fazer a análise estatística dos resultados**  
**\# Aplique técnicas estatísticas (ex.: teste t, intervalo de confiança, análise de variância) para comparar os modelos e validar a significância dos resultados.**  
**\# Gere graficos para as analises**  
**\# (6) Estimar o erro em dados não vistos (generalização)**  
**\# Combine os conjuntos de treino + validação para treinar o modelo.**  

**\# Use o conjunto de teste para estimar o erro final.**  
**\# Esse é o único ponto do processo em que o conjunto de teste deve ser utilizado — exclusivamente para avaliar a capacidade de generalização do modelo final.**  
**\# (7) Treinar a versão final do modelo**  
**\# Utilize todos os dados disponíveis (treino + validação + teste) para treinar a versão definitiva do modelo que será entregue ou implantada.**  
**\# (8) Elaborar o relatório do experimento**  
**\# Documente todo o processo, incluindo:**  
**\# População-alvo**  
**\# Processo de amostragem**  
**\# Pré-processamento**
**\#Configurações dos experimentos**  
**\# Justificativas para as escolhas**
**\#Resultados**
**\#Análise estatística**
**\#Limitações e próximos passos**

**Estudo Exploratório** 

Esse trabalho deve ser feito pelo em grupo de 4 alunos. Cada grupo deve então executar o estudo comparativo de 4 modelos de aprendizado supervisionado sobre um dataset de escolha do grupo.. Por se tratar de um estudo exploratório, caminhos serão seguidos e abandonados.Isso faz parte do percurso. Dos caminhos que se mostrarem potencialmente frutı́feros, escolha um para apresentar aqui. 

Escreva então um artigo relatando seus experimentos, contendo não mais de 8 páginas (sem contar referências), conforme o formato utilizado pela ACL 2023.

Este artigo deve abordar,pelo menos, os seguintes tópicos:

**Introdução ao estudo:** apresente a questão de pesquisa inicial, argumentando sobre por que ela é importante. Descreva brevemente a base de dados utilizada, apresentando um spoiler do que você fez e dos principais resultados.

Exploração: descreva como foi feita a exploração (focando no caminho escolhido). Aborde, mais especificamente:

- **Base de dados:** descreva criticamente a base de dados utilizada, apontando seu tamanho e como foi dividida para exploração inicial e para a execução das técnicas de aprendizado. Discuta os problemas encontrados com os dados, como duplicações, erros, incoerências ou ausências. É crucial a análise crı́tica da base, apontando suas limitações e potenciais viéses. Para isso, descreva também como a base foi obtida (ou seja, como foi amostrada por quem a criou).

- **Execução da exploração:** faça um breve relato da exploração, ilustrando as descobertas relevantes por meio de gráficos adequados aos tipos de dados analisados, bem elaborados, que ajudem realmente a ver esses resultados. Apresente então a questão refinada, questão esta alvo do seu experimento com aprendizado de máquina.  aqui a questão deve incluir explicitamente: essa comparação (e análise do desempenho). Para isso, é importante que se estabeleça uma argumentação sobre que fatores descobertos na exploração subsidiam a questão proposta.

**Materiais e métodos:**

- Base de dados: descreva onde os experimentos serão executados, como a base será dividida etc. Apontar que se usará a mesma base não é redundância óbvia.

- Técnicas de aprendizado: descreva também, de forma sucinta, as técnicas de aprendizado aplicadas, justificando sua escolha.

- Aplicação das técnicas: quando descrever como essas técnicas foram aplicadas aos dados, é crucial o fornecimento dos valores dos hiperparâmetros (se definidos de antemão), ou da metodologia para sua definição. Também é crucial a apresentação do valor de sementes para geradores aleatórios, bem como de que bibliotecas foram usadas para qual finalidade. Aqui o foco é a reprodutibilidade, e cabe a você descrever como seus experimentos de aprendizado foram executados e como podem ser reproduzidos. Basicamente, escreva algo que você gostaria de encontrar em um artigo que quisesse reproduzir.

- Avaliação dos modelos: descreva como os modelos serão avaliados individualmente (que medidas de avaliação foram usadas), justificando suas escolhas. Descreva também como os modelos serão comparados, e como você irá avaliar se as diferenças observadas são provavelmente reais e não resultados espúrios.

**Resultados:**

Descreva como o experimento foi executado, em que ambiente (software e hardware), quanto tempo durou a execução etc. Apresente então os resultados obtidos, ilustrando-os por meio de gráficos que ajudem o leitor a compreender suas conclusões.

Quando da comparação dos modelos, tenha muito cuidado com qualquer afirmação feita. Sempre se pergunte se você consegue fazê-la com um certo grau de certeza. Para isso, toda afirmação deve ser justificada, se não por algum teste estatı́stico relevante, pelo menos por uma análise que corrobore com a afirmação feita.

**Discussão dos resultados:** mostre que conclusões podem ser tiradas do seu experimento, sempre deixando claro seu grau de certeza (ainda que subjetivo) com relação a elas. Aponte limitações (e potenciais viéses) dos resultados, suas causas, e possı́veis consequências, bem como as questões que não foram respondidas. Aponte então direções a serem seguidas futuramente, para que essas limitações sejam sanadas, ou para que questões que ficaram sem resposta sejam abordadas. É importante lembrar que os tópicos acima não necessariamente correspondem a seções de seu documento. Para essas, você pode tanto unir quanto separar tópicos. A decisão é sua.

Note também que a limitação no tamanho do artigo força uma escolha editorial: o que entra ou não nele. Foquem em um ponto que queiram mostrar e sigam de forma coerente em sua direção. Quaisquer desvios interessantes, mas não explorados aqui, devem constar das direções de pesquisa futura.

**Material para Entrega**

A entrega será feita unica e exclusivamente via moodle. Você deve criar um arquivo

“Numero da matricula.pdf” (em que Numero da matricula  é seu registro de matrícula no IFSP) contendo o artigo gerado.

Deve ser submetido um único arquivo por grupo (ou seja, um único integrante irá fazer a submissão), sendo que o documento do planejamento deverá conter o nome e número de matricula de todos os participantes.

**Data de Entrega**

O prazo de entrega é 30/11/2025.

**Critérios de Avaliação**

Para a avaliação, serão observados os seguintes critérios:

- O cumprimento dos pontos elencados na Seção 1;

- A organização geral do documento, em especial, a coerência em torno do ponto que se des