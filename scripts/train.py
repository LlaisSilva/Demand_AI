import pandas as pd
from config import MODEL_PATH
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from data_utils import load_data, prepare_data, split_data, encode_features



def main():
    df = prepare_data(load_data())
    train_df, val_df, test_df = split_data(df)
    x_train = encode_features(train_df)
    x_val = encode_features(val_df, x_train.columns)
    y_train = train_df['NUM_ORDERS']
    y_val = val_df['NUM_ORDERS']

    baseline_means = train_df.groupby(['MEAL_ID', 'DAY_OF_WEEK'])['NUM_ORDERS'].mean()
    baseline_predictions = [baseline_means.loc[(meal, weekday)] for meal, weekday in zip(val_df['MEAL_ID'], val_df['DAY_OF_WEEK'])]

    print('Baseline validation MAE:', round(mean_absolute_error(y_val, baseline_predictions), 2))

    model = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=32, n_jobs=-1)
    model.fit(x_train, y_train)

    print('Random Forest validation MAE:', round(mean_absolute_error(y_val, model.predict(x_val)), 2))

    x_full = encode_features(pd.concat([train_df, val_df]), x_train.columns)
    y_full = pd.concat([train_df['NUM_ORDERS'], val_df['NUM_ORDERS']])

    model.fit(x_full, y_full)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump({'model': model, 'columns': list(x_train.columns)}, MODEL_PATH)
    print('Saved model:', MODEL_PATH)
    print('Test set reserved; no test metrics calculated during training.')


if __name__ == '__main__':
    main()
