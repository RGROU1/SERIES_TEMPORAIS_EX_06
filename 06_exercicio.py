import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import AutoARIMA, Prophet as DartsProphet, RegressionModel
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import torch
import pmdarima
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

# Configurações de estilo do Matplotlib
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['text.color'] = 'black'
plt.rcParams['grid.color'] = 'lightgray'

# Reprodutibilidade
np.random.seed(42)
torch.manual_seed(42)

# --- 1. Carregar e Preparar os Dados ---
try:
    df = pd.read_csv('dados_amostra.csv', header=None, names=['time_step', 'value'])
except FileNotFoundError:
    print("Erro: 'dados_amostra.csv' não encontrado.")
    exit()

# Para Prophet (Darts), é necessário um índice de data.
df_prophet = df.copy()
df_prophet['time_step_dt'] = pd.date_range(end=pd.Timestamp.today().normalize(), periods=len(df_prophet), freq='D')
series_datetime_index = TimeSeries.from_dataframe(df_prophet, 'time_step_dt', 'value', fill_missing_dates=True, freq='D')

# Para outros modelos (ARIMA, XGBoost), usamos o índice numérico original.
series_numeric_index = TimeSeries.from_dataframe(df, time_col='time_step', value_cols='value')

plt.figure(figsize=(12, 6))
series_numeric_index.plot(label='Série Original')
plt.title('Série Temporal Completa')
plt.legend()
plt.show()

# --- 2. Divisão Treino-Validação ---
N_PRED_FINAL = 18
N_VAL_INTERNA = 36
serie_treino_val_numeric = series_numeric_index
serie_treino_val_datetime = series_datetime_index

treino_interno_numeric, val_interna_numeric = serie_treino_val_numeric[:-N_VAL_INTERNA], serie_treino_val_numeric[-N_VAL_INTERNA:]
treino_interno_datetime, val_interna_datetime = serie_treino_val_datetime[:-N_VAL_INTERNA], serie_treino_val_datetime[-N_VAL_INTERNA:]

print(f"Comprimento da série completa fornecida: {len(serie_treino_val_numeric)}")
print(f"Comprimento do conjunto de treino interno: {len(treino_interno_numeric)}")
print(f"Comprimento do conjunto de validação interna: {len(val_interna_numeric)}")

# --- Função RMSE ---
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

desempenho_modelos = {}

# --- 3. Ajuste e Avaliação de Modelos ---

# 3.a AutoARIMA (Darts)
print("\n--- Treinando AutoARIMA (Darts) ---")
try:
    modelo_auto_arima_darts = AutoARIMA(random_state=42)
    modelo_auto_arima_darts.fit(treino_interno_numeric)
    pred_auto_arima_darts = modelo_auto_arima_darts.predict(n=N_VAL_INTERNA)
    rmse_auto_arima_darts = rmse(val_interna_numeric.values(), pred_auto_arima_darts.values())
    desempenho_modelos['AutoARIMA_Darts'] = rmse_auto_arima_darts
    print(f"AutoARIMA (Darts) RMSE: {rmse_auto_arima_darts:.4f}")
except Exception as e:
    print(f"AutoARIMA (Darts) falhou: {e}")
    desempenho_modelos['AutoARIMA_Darts'] = float('inf')

# 3.b Prophet (Darts)
print("\n--- Treinando Prophet (Darts) ---")
try:
    modelo_prophet_darts = DartsProphet()
    modelo_prophet_darts.fit(treino_interno_datetime)
    pred_prophet_darts = modelo_prophet_darts.predict(n=N_VAL_INTERNA)
    rmse_prophet_darts = rmse(val_interna_numeric.values(), pred_prophet_darts.values())
    desempenho_modelos['Prophet_Darts'] = rmse_prophet_darts
    print(f"Prophet (Darts) RMSE: {rmse_prophet_darts:.4f}")
except Exception as e:
    print(f"Prophet (Darts) falhou: {e}")
    desempenho_modelos['Prophet_Darts'] = float('inf')

# 3.c XGBoost (Darts RegressionModel)
OUTPUT_CHUNK_XGB = N_VAL_INTERNA
if len(treino_interno_numeric) > 72:
    INPUT_LAGS_XGB = 36
else:
    INPUT_LAGS_XGB = max(1, len(treino_interno_numeric) // 3)

print(f"Usando INPUT_LAGS_XGB={INPUT_LAGS_XGB}, OUTPUT_CHUNK_XGB={OUTPUT_CHUNK_XGB} para XGBoost.")

if INPUT_LAGS_XGB <= 0:
    print("INPUT_LAGS_XGB é zero ou negativo. Pulando XGBoost.")
    desempenho_modelos['XGBoost_Darts'] = float('inf')
else:
    print("\n--- Treinando XGBoost (Darts) ---")
    try:
        modelo_xgboost_darts = RegressionModel(
            lags=INPUT_LAGS_XGB,
            output_chunk_length=OUTPUT_CHUNK_XGB,
            model=xgb.XGBRegressor(random_state=42, n_estimators=100)
        )
        modelo_xgboost_darts.fit(treino_interno_numeric)
        pred_xgboost_darts = modelo_xgboost_darts.predict(n=N_VAL_INTERNA)
        rmse_xgboost_darts = rmse(val_interna_numeric.values(), pred_xgboost_darts.values())
        desempenho_modelos['XGBoost_Darts'] = rmse_xgboost_darts
        print(f"XGBoost (Darts) RMSE: {rmse_xgboost_darts:.4f}")
    except Exception as e:
        print(f"XGBoost (Darts) falhou: {e}")
        desempenho_modelos['XGBoost_Darts'] = float('inf')

# --- 4. Escolher o Melhor Modelo ---
print("\n--- Desempenho dos Modelos (RMSE na validação interna) ---")
for nome_modelo, score_rmse in desempenho_modelos.items():
    print(f"{nome_modelo}: {score_rmse:.4f}")

modelos_validos = {k: v for k, v in desempenho_modelos.items() if v is not None and math.isfinite(v)}
if not modelos_validos:
    print("Nenhum modelo foi treinado com sucesso. Não é possível selecionar o melhor modelo.")
    exit()

melhor_modelo_nome = min(modelos_validos, key=modelos_validos.get)
print(f"\nMelhor modelo com base na validação interna: {melhor_modelo_nome} com RMSE: {desempenho_modelos[melhor_modelo_nome]:.4f}")

# --- 5. Previsão Final ---
N_PRED_FINAL = 18
previsoes_finais = None

# Parâmetros para XGBoost no retreino final (se ele for o melhor)
if len(serie_treino_val_numeric) > 3 * N_PRED_FINAL:
    FINAL_INPUT_LAGS_XGB = 2 * N_PRED_FINAL
else:
    available_for_input_final = len(serie_treino_val_numeric) - N_PRED_FINAL
    FINAL_INPUT_LAGS_XGB = max(1, available_for_input_final // 2 if available_for_input_final > 1 else 0)
FINAL_OUTPUT_CHUNK_XGB = N_PRED_FINAL

print(f"\n--- Retreinando {melhor_modelo_nome} no dataset completo e prevendo {N_PRED_FINAL} passos ---")
if melhor_modelo_nome == 'XGBoost_Darts':
    print(f"Usando FINAL_INPUT_LAGS_XGB={FINAL_INPUT_LAGS_XGB}, FINAL_OUTPUT_CHUNK_XGB={FINAL_OUTPUT_CHUNK_XGB} para XGBoost final.")

modelo_final = None

if melhor_modelo_nome == 'AutoARIMA_Darts':
    modelo_final = AutoARIMA(random_state=42)
    modelo_final.fit(serie_treino_val_numeric)
    previsoes_finais = modelo_final.predict(n=N_PRED_FINAL)
elif melhor_modelo_nome == 'Prophet_Darts':
    modelo_final = DartsProphet()
    modelo_final.fit(serie_treino_val_datetime)
    previsoes_finais = modelo_final.predict(n=N_PRED_FINAL)
elif melhor_modelo_nome == 'XGBoost_Darts':
    lags_xgb_final = min(FINAL_INPUT_LAGS_XGB if FINAL_INPUT_LAGS_XGB > 0 else 36, 36)
    lags_xgb_final = max(1, lags_xgb_final)
    modelo_final = RegressionModel(
        lags=lags_xgb_final,
        output_chunk_length=FINAL_OUTPUT_CHUNK_XGB,
        model=xgb.XGBRegressor(random_state=42, n_estimators=100)
    )
    modelo_final.fit(serie_treino_val_numeric)
    previsoes_finais = modelo_final.predict(n=N_PRED_FINAL)

if previsoes_finais is not None:
    print("\nPrevisões Finais:")
    previsoes_array = previsoes_finais.values().flatten()
    print(previsoes_array)

    plt.figure(figsize=(12,6))
    serie_treino_val_numeric.plot(label='Série Completa Fornecida')
    previsoes_finais.plot(label=f'{melhor_modelo_nome} Previsão Final ({N_PRED_FINAL} passos)')
    plt.title(f'Previsão Final com {melhor_modelo_nome}')
    plt.legend()
    plt.show()

    # --- 6. Exportar Previsões ---
    df_previsoes_para_exportar = pd.DataFrame(previsoes_array)
    try:
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        caminho_exportacao = os.path.join(desktop_path, 'predictions_darts.csv')
        df_previsoes_para_exportar.to_csv(caminho_exportacao, index=False, header=False, decimal=',', sep=';')
        print(f"\nPrevisões exportadas para {caminho_exportacao}")
        print("Conteúdo do CSV (primeiras linhas):")
        print(df_previsoes_para_exportar.head())
    except Exception as e:
        print(f"Erro ao exportar CSV: {e}")
else:
    print("Nenhum modelo foi selecionado ou treinado com sucesso para a previsão final.")

# --- 7. Análise Clássica (opcional, para estudo) ---

# Gráfico da série temporal original
plt.figure(figsize=(12, 6))
plt.plot(df['time_step'], df['value'])
plt.title('Série Temporal Original')
plt.xlabel('Time Step')
plt.ylabel('Valor')
plt.grid(True)
plt.show()

# Decomposição ETS Aditiva
decomposition = seasonal_decompose(df['value'], model='additive', period=12)
fig = decomposition.plot()
fig.set_size_inches(12, 8)
plt.tight_layout()
plt.show()

# Gráficos ACF/PACF
fig, ax = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(df['value'].dropna(), lags=40, ax=ax[0])
plot_pacf(df['value'].dropna(), lags=40, ax=ax[1])
plt.tight_layout()
plt.show()

# Teste de Estacionariedade (ADF)
def adf_test(series):
    result = adfuller(series)
    print(f'ADF Statistic: {result[0]:.4f}')
    print(f'p-valor: {result[1]:.4f}')
    print('Valores Críticos:')
    for key, value in result[4].items():
        print(f'\t{key}: {value:.4f}')

print("\nTeste ADF para série original:")
adf_test(df['value'].dropna())

# Primeira diferença e nova análise
df['first_diff'] = df['value'].diff()

plt.figure(figsize=(12, 6))
plt.plot(df['time_step'][1:], df['first_diff'].iloc[1:])
plt.title('Primeira Diferença da Série')
plt.xlabel('Time Step')
plt.ylabel('Valor Diferenciado')
plt.grid(True)
plt.show()

fig, ax = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(df['first_diff'].dropna(), lags=40, ax=ax[0])
plot_pacf(df['first_diff'].dropna(), lags=40, ax=ax[1])
plt.tight_layout()
plt.show()

print("\nTeste ADF para primeira diferença:")
adf_test(df['first_diff'].dropna())

# Decomposição da série diferenciada
decomposition_diff = seasonal_decompose(df['first_diff'].dropna(), model='additive', period=12)
fig = decomposition_diff.plot()
fig.set_size_inches(12, 8)
plt.tight_layout()
plt.show()
