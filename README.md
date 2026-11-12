# DemandMind AI — Restaurant Demand Forecasting

DemandMind AI é um projeto de Machine Learning desenvolvido para prever a demanda de pratos em restaurantes, auxiliando no planejamento operacional e na gestão de ingredientes.

## Funcionalidades

- Previsão da quantidade de pedidos por prato e data.
- Identificação do prato mais pedido da semana.
- Ranking de pratos por demanda prevista.
- Comparação de demanda entre períodos.
- Identificação do dia com maior demanda.
- Estimativa de ingredientes necessários com base nas receitas cadastradas.

## Tecnologias

- Python
- Pandas
- Scikit-learn
- Random Forest
- Joblib
- Matplotlib e Seaborn

## Modelo

O projeto utiliza um modelo Random Forest Regressor treinado com dados históricos de vendas de restaurantes.

A avaliação utiliza divisão temporal dos dados em treinamento, validação e teste, com MAE (*Mean Absolute Error*) como métrica de desempenho.

## Como executar

1. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

2. Coloque o dataset em `data/data_preprocessed.csv`.

3. Execute `train.py` para treinar e salvar o modelo.

4. Execute `predictfeat: implement meal demand forecasting and analyticsfeat: implement meal demand forecasting and analytics.py` para gerar as previsões.

## Dataset

[Restaurant Sales Data with Features — Mendeley Data](https://data.mendeley.com/datasets/xv5kp6mxv7/1)

## Próximos passos

- Desenvolvimento de uma API REST com Java Spring Boot.
- Integração da API Java com o modelo Python.
- Gerenciamento de receitas, ingredientes e estoque.
