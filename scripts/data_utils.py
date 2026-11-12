from config import DATA_PATH
import pandas as pd

DISHES = ['CALAMARI', 'FISCH', 'GARNELEN', 'HAEHNCHEN', 'KOEFTE', 'LAMM', 'STEAK']

FEATURES = ['MEAL_ID', 'DAY_OF_WEEK', 'DAY', 'MONTH', 'ISHOLIDAY', 'WEEKEND']


def load_data():

    df = pd.read_csv(DATA_PATH, sep=';', index_col=0)

    df['DEMAND_DATE'] = pd.to_datetime(df['DEMAND_DATE'], format='%m.%d.%Y', errors='raise')

    return df.sort_values('DEMAND_DATE').reset_index(drop=True)


def prepare_data(df):

    df_model = df[['DEMAND_DATE', 'ISHOLIDAY', 'WEEKEND'] + DISHES].melt(
        id_vars=['DEMAND_DATE', 'ISHOLIDAY', 'WEEKEND'],
        value_vars=DISHES,
        var_name='MEAL_ID',
        value_name='NUM_ORDERS'
    )

    df_model['DAY_OF_WEEK'] = df_model['DEMAND_DATE'].dt.dayofweek
    df_model['DAY'] = df_model['DEMAND_DATE'].dt.day
    df_model['MONTH'] = df_model['DEMAND_DATE'].dt.month

    return df_model.sort_values(['DEMAND_DATE', 'MEAL_ID']).reset_index(drop=True)


def split_data(df_model):

    dates = df_model['DEMAND_DATE'].sort_values().unique()

    train_end = int(len(dates) * 0.70)
    val_end = int(len(dates) * 0.85)

    train = df_model[df_model['DEMAND_DATE'].isin(dates[:train_end])].copy()
    val = df_model[df_model['DEMAND_DATE'].isin(dates[train_end:val_end])].copy()
    test = df_model[df_model['DEMAND_DATE'].isin(dates[val_end:])].copy()

    return train, val, test


def encode_features(df, columns=None):

    x = pd.get_dummies(df[FEATURES].copy(), columns=['MEAL_ID'], dtype=int)

    if columns is not None:
        x = x.reindex(columns=columns, fill_value=0)

    return x
