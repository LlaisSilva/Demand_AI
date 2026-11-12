from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from data_utils import load_data, prepare_data, split_data, encode_features


def main():
    df = prepare_data(load_data())
    train_df, val_df, _ = split_data(df)
    x_train = encode_features(train_df)
    x_val = encode_features(val_df, x_train.columns)
    y_train = train_df['NUM_ORDERS']
    y_val = val_df['NUM_ORDERS']
    means = train_df.groupby(['MEAL_ID', 'DAY_OF_WEEK'])['NUM_ORDERS'].mean()
    baseline = [means.loc[(m, d)] for m, d in zip(val_df['MEAL_ID'], val_df['DAY_OF_WEEK'])]
    results = {'Baseline': mean_absolute_error(y_val, baseline)}

    models = {
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=32, n_jobs=-1),
        'Linear Regression': LinearRegression(),
    }
    try:
        from xgboost import XGBRegressor
        models['XGBoost'] = XGBRegressor(n_estimators=300, learning_rate=0.5, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=32, objective='reg:squarederror')
    except ImportError:
        print('XGBoost unavailable')
    try:
        from lightgbm import LGBMRegressor
        models['LightGBM'] = LGBMRegressor(n_estimators=200, learning_rate=0.03, max_depth=5, num_leaves=15, min_child_samples=20, random_state=42, verbosity=-1)
    except ImportError:
        print('LightGBM unavailable')

    for name, model in models.items():
        model.fit(x_train, y_train)
        results[name] = mean_absolute_error(y_val, model.predict(x_val))
    for name, mae in sorted(results.items(), key=lambda pair: pair[1]):
        print(f'{name} MAE: {mae:.2f}')


if __name__ == '__main__':
    main()
