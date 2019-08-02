import numpy as np
import pandas as pd
import scipy.sparse as sparse
import random
import copy
import dateutil.relativedelta
from implicit.nearest_neighbours import bm25_weight, tfidf_weight, normalize


def filter_old_cards(df, month_threshold=1, date_column='dates', card_column='CRD_NO'):
    """
    Removes old cards. Take those cards date of last purchase of which is less than max_date - month_threshold
    Args:
        df: Dataframe with bills
        month_threshold: Max month threshold of last purchase for specific crd
        date_column: column that contains sold date
        card_column: Column that contains crd number

    Returns:
    Filtered DataFrame
    """
    df[date_column] = pd.to_datetime(df[date_column])
    max_date = df[date_column].max()
    date_to_find = max_date - dateutil.relativedelta.relativedelta(months=month_threshold)
    cards_last_date = df.groupby([card_column])[date_column].max().reset_index(name='date_max')
    df = df.merge(cards_last_date, on=card_column, how='left')
    return df[df['date_max'] > date_to_find].drop('date_max', axis=1)


def filter_rare_cards(df, rarity_num=5, card_column='CRD_NO', date_column='dates'):
    """
    Removes rare cards. Takes only cards which have more than rarity_num bills (different days)
    Args:
        df: Dataframe with bills
        rarity_num: Minimum num bills per crd
        card_column: Column that contains crd number
        date_column: Column that contains date
    Returns:
    Filtered DataFrame
    """
    # drop_duplicates - Чтобы оставить только уникальные пары карта-день
    # (смотреть сколько разных дней покупатель приходил)
    cards_num_bills = df.drop_duplicates([date_column, card_column])\
                        .groupby([card_column]).size().reset_index(name='bill_counts')
    df = df.merge(cards_num_bills, on=card_column, how='left')
    return df[df['bill_counts'] > rarity_num].drop('bill_counts', axis=1)


def filter_rare_goods(df, rarity_num=5, date_column='dates', plu_column='PLU_ID'):
    """
    Removes rare goods. Takes only specific goods, num per month sold of which more than rarity num.
    Args:
        df: Dataframe with bills
        rarity_num: Minimum average num per month sold
        date_column: column that contains sold date
        plu_column: column that contains plu number

    Returns:
    Filtered DataFrame
    """
    df['year_month'] = pd.to_datetime(df[date_column]).dt.to_period('M')
    plu_num_bills = df.groupby([plu_column, 'year_month']).size().reset_index(name='plu_counts')
    plu_num_bills_per_month = plu_num_bills.groupby(plu_column)['plu_counts'].mean().reset_index(name='mean_num')
    df = df.merge(plu_num_bills_per_month, on=plu_column, how='left')
    return df[df['mean_num'] > rarity_num].drop(['mean_num'], axis=1)


def filter_old_goods(df, month_threshold=1, date_column='dates', plu_column='PLU_ID'):
    """
    Removes old goods. Takes only that goods, last date of sold of which is less than a max_date - month_threshold
    Args:
        df: Dataframe with bills
        month_threshold:  Max month threshold of last sale of specific plu
        date_column: column that contains sold date
        plu_column: column that contains plu number

    Returns:
    Filtered DataFrame
    """
    df[date_column] = pd.to_datetime(df[date_column])
    max_date = df[date_column].max()
    date_to_find = max_date - dateutil.relativedelta.relativedelta(months=month_threshold)
    plu_last_date = df.groupby([plu_column])[date_column].max().reset_index(name='date_max')
    df = df.merge(plu_last_date, on=plu_column, how='left')
    return df[df['date_max'] > date_to_find].drop('date_max', axis=1)


def filter_by_quantile(df, plu_count_quantiles=(0.5, 0.99), cards_count_quantiles=(0.4, 0.99),
                       plu_column='PLU_ID', card_column='CRD_NO'):
    # Cut df by plu_count
    """
    Filter by plu and card quantiles
    Args:
        df: DataFrame with bills
        plu_count_quantiles: list-like. Upper and lower quantile threshold of num appearances of plu
        cards_count_quantiles: list-like. Upper and lower quantile threshold of num appearances of crd
        plu_column: Column that contains PLU_ID
        card_column: Column that contains CRD_NO
    Returns:
    Filtered dataframe
    """
    df = df[(df.groupby(plu_column)[plu_column].transform('size') >
             df[plu_column].value_counts().quantile(plu_count_quantiles[0]))
            &
            (df.groupby(plu_column)[plu_column].transform('size') <
             df[plu_column].value_counts().quantile(plu_count_quantiles[1]))]

    # Cut df by cards_count
    df = df[(df.groupby([card_column])[card_column].transform('size') >
             df[card_column].value_counts().quantile(cards_count_quantiles[0]))
            &
            (df.groupby([card_column])[card_column].transform('size') <
             df[card_column].value_counts().quantile(cards_count_quantiles[1]))]
    return df


class Dataset:
    def __init__(self, df):
        self.df = df
        self.df = self.columns_to_lowercase(self.df)

    def make_matrix(self, plu_column='plu_id', card_column='crd_no'):
        """
        Args:
            plu_column: string. Name of column which contains PLU
            card_column: string. Name of column which contains Card numbers
        Returns:
            df: dataframe
        """
        return self.df.reset_index()\
                      .pivot_table(columns=[plu_column], index=[card_column], values='index', aggfunc='count')\
                      .fillna(0)
        
    @staticmethod
    def transform(matrix, method='no', clip_upper_value=100):
        """
        Function transforms every single value in matrix with specified rules
        Args:
            matrix: Matrix to transform
            method: Transformation method (no, clip)
            clip_upper_value: clip upper value

        Returns:
            Transformed matrix
        """
        if method == 'no':
            return matrix
        elif method == 'clip':
            return matrix.clip(upper=clip_upper_value)
        elif method == 'log':
            return matrix.apply(np.log).clip(0, clip_upper_value)

    @staticmethod
    def columns_to_lowercase(df):
        df.columns = [x.lower() for x in df.columns]
        return df

    @staticmethod
    def apply_weights(df, weight='bm25'):
        """
        Function apply weights to user-item matrix
        Args:
            df: Matrix user-item
            weight: (bm25, tf-idf, normalize) - weight method

        Returns:
        Weighted user-item matrix
        """
        if weight == 'bm25':
            crd_list = list(df.index.values)
            plu_list = list(df.columns)
            matrix = pd.DataFrame(bm25_weight(sparse.csr_matrix(df.to_numpy(), dtype='float16'), B=0.9).toarray())
            matrix.columns = plu_list
            matrix.index = crd_list
            return matrix

        if weight == 'tf-idf':
            crd_list = list(df.index.values)
            plu_list = list(df.columns)
            matrix = pd.DataFrame(tfidf_weight(sparse.csr_matrix(df.to_numpy(), dtype='float16')).toarray())
            matrix.columns = plu_list
            matrix.index = crd_list
            return matrix

        if weight == 'normalize':
            crd_list = list(df.index.values)
            plu_list = list(df.columns)
            matrix = pd.DataFrame(normalize(sparse.csr_matrix(df.to_numpy(), dtype='float16')).toarray())
            matrix.columns = plu_list
            matrix.index = crd_list
            return matrix
