
# Previsão de Série Temporal trabalho intermediário ex 06

Este projeto tem como objetivo ajustar e avaliar diferentes modelos de séries temporais para prever as próximas 18 observações de uma série temporal fornecida no arquivo `dados_amostra.csv`. O desempenho dos modelos é avaliado com base na métrica RMSE (Erro Quadrático Médio Raiz).

## Estrutura do Projeto

-   **`dados_amostra.csv`**: Arquivo CSV contendo a série temporal original. A primeira coluna é um índice temporal numérico e a segunda coluna contém os valores da série.
-   **`script_previsao.py`** (ou o nome do seu script Python): Contém o código Python para carregar os dados, treinar os modelos, avaliar o desempenho e gerar as previsões finais.
-   **`predictions_darts.csv`**: Arquivo CSV gerado pelo script, contendo as 18 previsões finais (um valor por linha, sem cabeçalho, decimal com vírgula, separador com ponto e vírgula).
-   **`README.md`**: Este arquivo.

## Modelos Utilizados

Foram implementados e avaliados os seguintes modelos utilizando a biblioteca Darts:

1.  **AutoARIMA (Darts)**: Utiliza o `pmdarima.auto_arima` para encontrar automaticamente as melhores ordens (p,d,q)(P,D,Q,m) para o modelo ARIMA/SARIMA.
2.  **Prophet (Darts)**: Modelo de previsão da Meta, adequado para séries com tendências e sazonalidades.
3.  **XGBoost (Darts RegressionModel)**: Modelo de gradient boosting, onde o problema de série temporal é transformado em um problema de regressão utilizando lags (valores passados) como features.

O modelo Transformer foi inicialmente considerado, mas removido da análise final devido a instabilidades na execução com o conjunto de dados fornecido.

## Metodologia

1.  **Carregamento e Preparação dos Dados**:
    *   A série temporal é carregada do `dados_amostra.csv`.
    *   Para o Prophet, um índice temporal do tipo `datetime` fictício é criado. Para os demais modelos, utiliza-se o índice numérico original.

2.  **Divisão Treino-Validação**:
    *   A série fornecida (1062 observações) é dividida em:
        *   **Conjunto de Treino Interno**: As primeiras 1026 observações.
        *   **Conjunto de Validação Interna**: As últimas 36 observações.
    *   Esta divisão é usada para selecionar o melhor modelo antes de fazer a previsão final.

3.  **Métrica de Avaliação**:
    *   O **RMSE** (Erro Quadrático Médio Raiz) é usado para comparar o desempenho dos modelos no conjunto de validação interna.

4.  **Ajuste e Seleção do Modelo**:
    *   Cada modelo (AutoARIMA, Prophet, XGBoost) é treinado no conjunto de treino interno.
    *   As previsões são feitas para os 36 períodos do conjunto de validação interna.
    *   O modelo com o menor RMSE na validação interna é selecionado como o melhor modelo.

5.  **Previsão Final**:
    *   O melhor modelo selecionado é **retreinado** utilizando **toda a série temporal fornecida** (as 1062 observações).
    *   Este modelo retreinado é então usado para prever as próximas 18 observações futuras.

6.  **Exportação**:
    *   As 18 previsões finais são salvas no arquivo `predictions_darts.csv`.

## Resultados da Validação Interna (Exemplo)

No exemplo de execução, os seguintes RMSEs foram obtidos no conjunto de validação interna:

*   **AutoARIMA (Darts)**: 266.3846
*   **Prophet (Darts)**: 734.4693
*   **XGBoost (Darts)**: 454.2741

Com base nestes resultados, o **AutoARIMA (Darts)** foi selecionado como o melhor modelo.

**Modelo AutoARIMA Selecionado (Exemplo do Sumário `pmdarima`):**

Isso indica um modelo ARIMA não sazonal com 2 termos autorregressivos, 1 ordem de diferenciação e 2 termos de média móvel.

## Como Executar

1.  **Pré-requisitos**:
    *   Python 3.8 ou superior.
    *   As seguintes bibliotecas Python instaladas:
        *   `pandas`
        *   `numpy`
        *   `matplotlib`
        *   `u8darts` (que inclui `torch`)
        *   `pmdarima` (para AutoARIMA)
        *   `prophet` (para o modelo Prophet)
        *   `xgboost`
        *   `scikit-learn`

    Você pode instalar as dependências principais com pip:
    ```bash
    pip install pandas numpy matplotlib u8darts pmdarima prophet xgboost scikit-learn
    ```
    *Nota: A instalação do Prophet pode ter dependências adicionais como o `cmdstanpy`. Siga as instruções de instalação da Darts/Prophet se encontrar problemas.*

2.  **Estrutura de Arquivos**:
    *   Coloque o arquivo `dados_amostra.csv` no mesmo diretório do script Python (`script_previsao.py`).

3.  **Execução**:
    *   Execute o script Python:
        ```bash
        python script_previsao.py
        ```

4.  **Saída**:
    *   O script imprimirá o RMSE de cada modelo na validação interna e o nome do melhor modelo selecionado.
    *   Um gráfico da série original e das previsões finais será exibido.
    *   O arquivo `predictions_darts.csv` será gerado na sua Área de Trabalho (Desktop) contendo as 18 previsões.

## Considerações

*   A performance dos modelos pode variar com diferentes sementes aleatórias (embora tenham sido fixadas para reprodutibilidade) e, principalmente, com um ajuste fino mais extenso de hiperparâmetros.
*   Para o modelo Prophet, um índice de data fictício foi criado. Se a periodicidade real da série fosse conhecida (diária, horária, etc.), o Prophet poderia se beneficiar dessa informação.
*   O modelo Transformer foi omitido devido a dificuldades de ajuste com o tamanho atual da série, mas poderia ser uma opção para séries temporais mais longas e complexas.