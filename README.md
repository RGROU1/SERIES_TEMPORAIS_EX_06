
# Exercício 06 – Análise de Predições com Modelos de Séries Temporais  

Este projeto compara modelos clássicos de séries temporais (AutoARIMA e Prophet) com um modelo de machine learning (XGBoost), aplicados à previsão de séries temporais utilizando a biblioteca Darts. O objetivo é avaliar o desempenho preditivo e gerar previsões futuras com base em dados históricos.
A escolha da biblioteca Darts para séries temporais é justificada por características que a tornam mais versátil e eficiente que alternativas como `pmdarima`, `Prophet` ou `Kats`

---

## Relatório Detalhado

Para uma análise completa, incluindo metodologia, métricas, gráficos e insights, consulte o [ Relatório Técnico](REPORT.md).

---

## Estrutura do Projeto

```bash
.
├── 06_exercicio.py     # Script principal da análise
├── predictions.csv     # Resultados de previsão gerados
├── requirements.txt    # Dependências do projeto (pip)
├── .gitignore          # Arquivos a serem ignorados pelo Git
├── REPORT.md           # Relatório técnico detalhado
└── README.md           # Documentação e instruções (este arquivo)
```

---

## Configuração do Ambiente

Para executar este projeto, siga os passos abaixo:

### 1. Clone o repositório

```bash
git clone https://github.com/SEU_USUARIO/NOME_DO_REPOSITORIO.git
cd NOME_DO_REPOSITORIO
```

### 2. Crie e ative um ambiente virtual (recomendado)

```bash
# Criar ambiente virtual
python -m venv venv
```

#### Ativar ambiente:
- **Windows**:
```bash
.env\Scriptsctivate
```
- **macOS/Linux**:
```bash
source venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

---

## ▶️ Como Executar

Após configurar o ambiente, execute o script principal com:

```bash
python 06_exercicio.py
```

---

## Etapas Executadas pelo Script

✔️ Carrega os dados históricos  
✔️ Aplica 3 modelos: AutoARIMA, Prophet, XGBoost  
✔️ Avalia o desempenho com RMSE  
✔️ Seleciona o melhor modelo  
✔️ Realiza a previsão de 30 períodos futuros  
✔️ Exporta os resultados para `predictions.csv`

---

## Tecnologias Utilizadas

- **[Darts](https://github.com/unit8co/darts)** – como a biblioteca Dart tem todos os modelos optamos por ela
- **Pandas** e **NumPy** – Manipulação e análise de dados
- **XGBoost** – Algoritmo de boosting supervisionado
- **Prophet** – Modelo aditivo sazonal da Meta
