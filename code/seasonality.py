import pandas as pd


def get_seasonality_weekly(bills, date_column='dates', group_column='level_4_name',
                           regular_only=False, promo_fact_column=None):
    bills['week'] = pd.to_datetime(bills[date_column]).dt.week
    bills['year'] = pd.to_datetime(bills[date_column]).dt.year
    # - Группируем по неделя-год, суммируем. Группируем по неделям, усредняем. (Если данные неравномерные)
    if not regular_only:
        num_per_week = bills.groupby([group_column, 'week', 'year'])[group_column].count().reset_index(name='num_sold')
        num_per_week = num_per_week.groupby([group_column, 'week'])['num_sold'].mean().reset_index(name='num_sold')
    else:
        # - Выбираем только регулярные продажи, считаем кол-во продаж и кол-во plu продававшихся регулярно на неделе
        num_per_week = bills[bills[promo_fact_column] == 0].groupby([group_column, 'week', 'year']).agg(
            {group_column: 'count', 'PLU_ID': 'nunique'})
        num_per_week = num_per_week.rename(columns={group_column: 'total_sold', 'PLU_ID': 'unique_plu'}).reset_index()
        # - Берем среднее по кол-ву рег. продаж и кол-ву рег. PLU по неделям между годами
        num_per_week = num_per_week.groupby([group_column, 'week'])[['total_sold', 'unique_plu']].mean().reset_index()
        # - Считаем кол-во регулярных продаж на кол-во рег. PLU (другими словами, если будет много товаров в категории
        # - На промо, то мы всё равно получим адекватную цифру.
        # - +10 - регуляризация
        num_per_week['num_sold'] = num_per_week['total_sold'] / (num_per_week['unique_plu']+10)
        num_per_week.drop(['total_sold', 'unique_plu'], axis=1, inplace=True)
    # - Делаем таблицу в которой есть все Категории и для каждого есть 52 недели
    new_table = pd.concat(
        [pd.DataFrame({group_column: x, 'week': [x + 1 for x in range(52)]}) for x in bills[group_column].unique()])
    # - Добавляем туда фактические продажи и если продаж нет то заполняем нулями
    new_table = new_table.merge(num_per_week, on=[group_column, 'week'], how='left').fillna(0)
    # - Добавляем общее кол-во проданных PLU за всё время
    total_sold = new_table.groupby([group_column])['num_sold'].sum().reset_index(name='total_sold')
    new_table = new_table.merge(total_sold, on=group_column, how='left')
    # - Добавляем кол-во проданных на следующей и предыдущей неделе
    new_table['num_sold_prev'] = new_table.sort_values('week').groupby([group_column]).num_sold.shift(1)
    new_table['num_sold_next'] = new_table.sort_values('week').groupby([group_column]).num_sold.shift(-1)
    # - Обрабатываем граничные условия (52 и 1 неделя года)
    plu_52_week_sales = dict(new_table[new_table['week'] == 52].set_index([group_column])['num_sold'])
    plu_1_week_sales = dict(new_table[new_table['week'] == 1].set_index([group_column])['num_sold'])
    new_table.loc[new_table['week'] == 1, 'num_sold_prev'] = new_table[new_table['week'] == 1][group_column].map(
        lambda x: plu_52_week_sales[x])
    new_table.loc[new_table['week'] == 52, 'num_sold_next'] = new_table[new_table['week'] == 52][group_column].map(
        lambda x: plu_1_week_sales[x])
    # - Считаем скользящее среднее
    new_table['rolling_average'] = (new_table['num_sold_prev'] + new_table['num_sold'] + new_table['num_sold_next']) / \
                                   (3 * new_table['total_sold'])
    return new_table[[group_column, 'week', 'rolling_average']]
