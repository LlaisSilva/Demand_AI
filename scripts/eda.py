import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_utils import load_data, DISHES


def main():

    df = load_data()

    print('Dimensions:', df.shape)
    print('Column types:')

    print(df.dtypes)

    print('Columns:', df.columns.tolist())
    print('Invalid dates:', df['DEMAND_DATE'].isna().sum())
    print('Duplicated rows:', df.duplicated().sum())
    print('Duplicated dates:', df['DEMAND_DATE'].duplicated().sum())
    print('Missing values:', df.isna().sum().sum())
    print('Start:', df['DEMAND_DATE'].min())
    print('End:', df['DEMAND_DATE'].max())
    print('Missing values by column:')

    print(df.isna().sum().loc[lambda x: x > 0].sort_values(ascending=False).head(20))

    print('Sales statistics:')
    print(df[DISHES].describe())
    print('Negative sales:')
    print((df[DISHES] < 0).sum())
    print('Zero sales:')
    print((df[DISHES] == 0).sum())

    print('Holidays:')
    print(df['ISHOLIDAY'].value_counts())

    dates = pd.date_range(df['DEMAND_DATE'].min(), df['DEMAND_DATE'].max(), freq='D')
    missing = dates.difference(pd.DatetimeIndex(df['DEMAND_DATE']))

    print('Calendar days:', len(dates))
    print('Days with entries:', len(df))
    print('Days with no entries:', len(missing))
    print(pd.DataFrame({'Missing date': missing}).head(20))

    plt.figure(figsize=(15, 6))

    for dish in DISHES:
        plt.plot(df['DEMAND_DATE'], df[dish].rolling(14, min_periods=1).mean(), label=dish)

    plt.title('Demand Trend by Dish')
    plt.xlabel('Date')
    plt.ylabel('Average of 14 entries')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    holiday_comparison = df.groupby('ISHOLIDAY')[DISHES].mean().T

    print('Holiday comparison:')
    print(holiday_comparison.round(2))

    holiday_comparison.plot(kind='bar', figsize=(12, 6))
    plt.title('Mean Demand by Dish: Holidays vs Regular Days')
    plt.xlabel('Dish')
    plt.ylabel('Average quantity sold')
    plt.tight_layout()
    plt.show()

    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df['DAY_OF_WEEK'] = df['DEMAND_DATE'].dt.day_name()

    weekly_demand = df.groupby('DAY_OF_WEEK')[DISHES].mean().reindex(weekday_order)

    print('Weekly demand:')
    print(weekly_demand.round(2))

    plt.figure(figsize=(12, 6))
    sns.heatmap(weekly_demand.T, annot=True, fmt='.1f', cmap='YlOrRd')
    plt.title('Average Demand by Dish and Day of the Week')
    plt.xlabel('Day of the week')
    plt.ylabel('Dish')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df[DISHES])
    plt.title('Distribution of Demand by Dish')
    plt.xlabel('Dish')
    plt.ylabel('Quantity Sold')
    plt.xticks(rotation=30)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
