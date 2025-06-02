# Previsão de Série Temporal - Trabalho Intermediário

Este projeto tem como objetivo ajustar e avaliar diferentes modelos de séries temporais para prever as próximas 18 observações de uma série temporal fornecida no arquivo `dados_amostra.csv`. O desempenho dos modelos é avaliado com base na métrica RMSE (Erro Quadrático Médio Raiz).

## Estrutura do Projeto

- **`dados_amostra.csv`**: Arquivo CSV contendo a série temporal original. A primeira coluna é um índice temporal numérico e a segunda coluna contém os valores da série.
- **`script_previsao.py`**: Contém o código Python para carregar os dados, treinar os modelos, avaliar o desempenho e gerar as previsões finais.
- **`predictions_darts.csv`**: Arquivo CSV gerado pelo script, contendo as 18 previsões finais (um valor por linha, sem cabeçalho, decimal com vírgula, separador com ponto e vírgula).
- **`README.md`**: Este arquivo.

## Modelos Utilizados

Foram implementados e avaliados os seguintes modelos utilizando a biblioteca Darts:

1. **AutoARIMA (Darts)**: Utiliza o `pmdarima.auto_arima` para encontrar automaticamente as melhores ordens (p,d,q)(P,D,Q,m) para o modelo ARIMA/SARIMA.
2. **Prophet (Darts)**: Modelo de previsão da Meta, adequado para séries com tendências e sazonalidades.
3. **XGBoost (Darts RegressionModel)**: Modelo de gradient boosting, onde o problema de série temporal é transformado em um problema de regressão utilizando lags (valores passados) como features.

O modelo Transformer foi inicialmente considerado, mas removido da análise final devido a instabilidades na execução com o conjunto de dados fornecido.

## Metodologia

1. **Carregamento e Preparação dos Dados**:
   - A série temporal é carregada do `dados_amostra.csv`.
   - Para o Prophet, um índice temporal do tipo `datetime` fictício é criado. Para os demais modelos, utiliza-se o índice numérico original.

2. **Divisão Treino-Validação**:
   - A série fornecida (1062 observações) é dividida em:
     - **Conjunto de Treino Interno**: As primeiras 1026 observações.
     - **Conjunto de Validação Interna**: As últimas 36 observações.
   - Esta divisão é usada para selecionar o melhor modelo antes de fazer a previsão final.

3. **Métrica de Avaliação**:
   - O **RMSE** (Erro Quadrático Médio Raiz) é usado para comparar o desempenho dos modelos no conjunto de validação interna.

4. **Ajuste e Seleção do Modelo**:
   - Cada modelo (AutoARIMA, Prophet, XGBoost) é treinado no conjunto de treino interno.
   - As previsões são feitas para os 36 períodos do conjunto de validação interna.
   - O modelo com o menor RMSE na validação interna é selecionado como o melhor modelo.

5. **Previsão Final**:
   - O melhor modelo selecionado é **retreinado** utilizando **toda a série temporal fornecida** (as 1062 observações).
   - Este modelo retreinado é então usado para prever as próximas 18 observações futuras.

---

## Outputs e Visualizações

A seguir, estão as principais imagens geradas durante a análise e modelagem. Todas as imagens são salvas na pasta `output_images/`:

### **1. Série Temporal Original**
![Série Temporal Original](output_images/serie_temporal_original.png)

### **2. Decomposição ETS Aditiva**
![Decomposição ETS Aditiva](output_images/decomposicao_ets_aditiva.png)

### **3. Gráficos ACF/PACF da Série Original**
![ACF/PACF Série Original](output_images/acf_pacf_serie_original.png)

### **4. Primeira Diferença da Série**
![Primeira Diferença da Série](output_images/primeira_diferenca_serie.png)

### **5. ACF/PACF da Primeira Diferença**
![ACF/PACF Primeira Diferença](output_images/acf_pacf_primeira_diferenca.png)

### **6. Decomposição da Primeira Diferença**
![Decomposição Primeira Diferença](output_images/decomposicao_primeira_diferenca.png)

> **Nota:** As imagens são geradas automaticamente pelo script e devem ser incluídas no repositório para visualização completa do relatório.

---