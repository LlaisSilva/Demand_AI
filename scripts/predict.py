
from config import MODEL_PATH
import joblib
import pandas as pd
from data_utils import DISHES, FEATURES


def predict_orders(dish, start_date, days=7, holiday_dates=None):
    dish = dish.upper()
    if dish not in DISHES:
        raise ValueError(f'Invalid dish. Choose one of: {", ".join(DISHES)}')
    if not 1 <= days <= 30:
        raise ValueError('days must be between 1 and 30')

    start = pd.Timestamp(start_date)
    holidays = {pd.Timestamp(d).normalize() for d in (holiday_dates or [])}
    dates = pd.date_range(start, periods=days, freq='D')

    data = pd.DataFrame({
        'MEAL_ID': dish,
        'DAY_OF_WEEK': dates.dayofweek,
        'DAY': dates.day,
        'MONTH': dates.month,
        'ISHOLIDAY': [int(d.normalize() in holidays) for d in dates],
        'WEEKEND': (dates.dayofweek >= 5).astype(int),
    })

    saved = joblib.load(MODEL_PATH)

    features = pd.get_dummies(data[FEATURES], columns=['MEAL_ID'], dtype=int)
    features = features.reindex(columns=saved['columns'], fill_value=0)

    predictions = saved['model'].predict(features)

    return pd.DataFrame({
        'DEMAND_DATE': dates,
        'MEAL_ID': dish,
        'PREDICTED_ORDERS': predictions.clip(min=0).round().astype(int)
    })


def predict_all_dishes(start_date, days=7, holiday_dates=None):
    return pd.concat(
        [predict_orders(dish, start_date, days, holiday_dates) for dish in DISHES],
        ignore_index=True
    ).sort_values(['DEMAND_DATE', 'MEAL_ID']).reset_index(drop=True)


def dishes_ranking(start_date, days=7, holiday_dates=None):
    predictions = predict_all_dishes(start_date, days, holiday_dates)

    ranking = (
        predictions.groupby('MEAL_ID', as_index=False)['PREDICTED_ORDERS']
        .sum()
        .sort_values(['PREDICTED_ORDERS', 'MEAL_ID'], ascending=[False, True])
        .reset_index(drop=True)
    )

    ranking.insert(0, 'RANK', range(1, len(ranking) + 1))

    return ranking


def most_ordered_dish(start_date, days=7, holiday_dates=None):
    return dishes_ranking(start_date, days, holiday_dates).iloc[0].to_dict()


def busiest_day(start_date, days=7, holiday_dates=None):
    predictions = predict_all_dishes(start_date, days, holiday_dates)

    daily = predictions.groupby('DEMAND_DATE', as_index=False)['PREDICTED_ORDERS'].sum()

    daily = daily.sort_values(
        ['PREDICTED_ORDERS', 'DEMAND_DATE'],
        ascending=[False, True]
    )

    return daily.iloc[0].to_dict()


def compare_periods(start_date_1, start_date_2, days=7, holiday_dates=None):
    first = dishes_ranking(
        start_date_1, days, holiday_dates
    )[['MEAL_ID', 'PREDICTED_ORDERS']]

    second = dishes_ranking(
        start_date_2, days, holiday_dates
    )[['MEAL_ID', 'PREDICTED_ORDERS']]

    result = first.merge(
        second,
        on='MEAL_ID',
        suffixes=('_PERIOD_1', '_PERIOD_2')
    )

    result['DIFFERENCE'] = (
        result['PREDICTED_ORDERS_PERIOD_2']
        - result['PREDICTED_ORDERS_PERIOD_1']
    )

    original = result['PREDICTED_ORDERS_PERIOD_1']

    result['CHANGE_PERCENT'] = (
        result['DIFFERENCE'] / original.where(original != 0) * 100
    ).round(2)

    return result.sort_values(
        'PREDICTED_ORDERS_PERIOD_2',
        ascending=False
    ).reset_index(drop=True)


def plan_ingredients(start_date, days=7, recipes=None, holiday_dates=None):
    if not isinstance(recipes, dict) or not recipes:
        raise ValueError(
            'Provide recipes as {dish: {ingredient: quantity_per_order}}'
        )

    unknown = set(recipes) - set(DISHES)

    if unknown:
        raise ValueError(f'Invalid dishes in recipes: {sorted(unknown)}')

    ranking = dishes_ranking(start_date, days, holiday_dates)

    orders = dict(zip(
        ranking['MEAL_ID'],
        ranking['PREDICTED_ORDERS']
    ))

    totals = {}

    for dish, ingredients in recipes.items():
        if not isinstance(ingredients, dict):
            raise ValueError(f'Ingredients for {dish} must be a dictionary')

        for ingredient, quantity in ingredients.items():
            if (
                not isinstance(quantity, (int, float))
                or not __import__("math").isfinite(quantity)
                or quantity < 0
            ):
                raise ValueError(
                    f'Invalid quantity for {ingredient} in {dish}'
                )

            totals[ingredient] = (
                totals.get(ingredient, 0)
                + orders[dish] * quantity
            )

    return pd.DataFrame(
        [
            {
                'INGREDIENT': ingredient,
                'REQUIRED_QUANTITY': quantity
            }
            for ingredient, quantity in sorted(totals.items())
        ]
    )



if __name__ == '__main__':
    print("PREVISAO DE TODOS OS PRATOS")
    print(predict_all_dishes("2015-11-08", 7))

    print("\nPRATO MAIS PEDIDO")
    print(most_ordered_dish("2015-11-08", 7))

    print("\nRANKING DE PRATOS")
    print(dishes_ranking("2015-11-08", 7))

    print("\nDIA COM MAIOR DEMANDA")
    print(busiest_day("2015-11-08", 7))

    print("\nCOMPARACAO ENTRE SEMANAS")
    print(compare_periods("2015-11-08", "2015-11-15", 7))

    recipes = {
        "STEAK": {
            "Carne (kg)": 0.2,
            "Batata (kg)": 0.15
        },
        "FISCH": {
            "Peixe (kg)": 0.25,
            "Batata (kg)": 0.1
        }
    }

    print("\nPLANEJAMENTO DE INGREDIENTES")
    print(plan_ingredients("2015-11-08", 7, recipes))

