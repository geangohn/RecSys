import pandas as pd
import numpy as np

class Baseline():
    """Обучение бейзлайнов
    
    baseline:
    - 'top_N'
    - 'top_N_different_cat'
    """
    
    def __init__(self, 
                 products,
                 user_item,
                 ctm_fact_column,
                 category_column,
                 item_name_column,
                 baseline = 'top_N_different_cat',
                 stop_words = ['Кулич', 'кулич', 'ПАСХА', 'пасха']):
        
        self.baseline_ = baseline
        self.user_item_ = user_item
        self.products_ = products
        self.ctm_fact_column_ = ctm_fact_column
        self.category_column_ = category_column
        self.item_name_column_ = item_name_column
        self.stop_words_ = stop_words
        
    
    def fit(self, train):

        data_for_dummy = train.groupby(['plu_id', 'dates']).count().reset_index().rename(columns = {'crd_no': 'n_purchases'})
        data_for_dummy = data_for_dummy[['plu_id', 'n_purchases']].groupby('plu_id').mean().reset_index()
        data_for_dummy = data_for_dummy.merge(self.products_[['plu_id', self.item_name_column_, self.category_column_, self.ctm_fact_column_]], on = 'plu_id', how = 'inner')
        data_for_dummy = data_for_dummy[data_for_dummy[self.ctm_fact_column_].values == 1]
        data_for_dummy.sort_values('n_purchases', ascending = False, inplace = True)

        # apply stop words
        for stop_word in self.stop_words_:
            data_for_dummy = data_for_dummy[[stop_word not in plu_name for plu_name in data_for_dummy[self.item_name_column_]]]
            
        self.data_for_dummy_ = data_for_dummy
            
    
    def recommend(self, userid, user_items = None, filter_already_liked_items = False, N = 5):
        def max_value(df):
            m = max(df.n_purchases)
            return df[df.n_purchases == m]

        if self.baseline_ == 'top_N':
            recs = self.data_for_dummy_.plu_id[:N].values 
        if self.baseline_ == 'top_N_different_cat':
            recs = self.data_for_dummy_.groupby(self.category_column_).apply(max_value).sort_values('n_purchases', ascending = False).plu_id[:N].values
            
        return [[item, 1] for item in recs]
