import numpy as np
import pandas as pd
import scipy.sparse as sparse
import random
import copy
from multiprocessing import Pool
from functools import partial
from sklearn.metrics.pairwise import cosine_similarity


# метрика на 1 карту
def fuzzy_crd(rec_ids, fact_ids, item_factors, plu_category_dict, weight_by_price=True, recs_prices_list=None):
    if (len(rec_ids) != 0) & (len(fact_ids) != 0):
        similarity_matrix = get_cosine_similarity(fact_ids, rec_ids, item_factors)
        heaviside_matrix = get_heaviside_matrix(fact_ids, rec_ids, plu_category_dict)
        fzy_col = np.amax(similarity_matrix * heaviside_matrix.T, axis=1)
        
        if weight_by_price:
            fzy_col_w = [f*price for f, price in zip(fzy_col, recs_prices_list)]
            fzy = sum(fzy_col_w) / sum(recs_prices_list)
        else:
            fzy = sum(fzy_col) / len(fzy_col)
        return fzy
    else:
        return np.nan
    
    
def precision_crd(rec_ids, fact_ids, weight_by_price=True, recs_prices_list=None):
    if (len(rec_ids) != 0) & (len(fact_ids) != 0):
        if weight_by_price:
            guessed = [rec in fact_ids for rec in rec_ids]
            guessed_w = [rec*price for rec, price in zip(guessed, recs_prices_list)]
            precision_score = sum(guessed_w) / sum(recs_prices_list)
        else:
            guessed = [rec in fact_ids for rec in rec_ids]
            precision_score = sum(guessed) / len(fact_ids)
        return precision_score
    else:
        return np.nan


def get_heaviside_matrix(fact_ids, rec_ids, plu_category_dict):
    heaviside_matrix = []
    for fct in fact_ids:
        dim = []
        for rcm in rec_ids:
            dim.append(heaviside(plu_category_dict[fct], plu_category_dict[rcm]))
        heaviside_matrix.append(dim)
    return np.array(heaviside_matrix)


def heaviside(cat1, cat2):
    if cat1 == cat2:
        return 1
    else:
        return 0


def get_cosine_similarity(id1, id2, item_factors):
    factors_1_full = []
    factors_2_full = []
    for id_ in id1:
        factors_1 = item_factors.loc[id_].tolist()
        factors_1_full.append(factors_1)
    for id_ in id2:
        factors_2 = item_factors.loc[id_].tolist()
        factors_2_full.append(factors_2)
    return cosine_similarity(factors_2_full, factors_1_full)


def get_recommendations(model, user_items_sparse_matrix, index_crd_dict, filter_already_liked, num_recommendations):
    recs = model.recommend_all(user_items=sparse.csr_matrix(user_items_sparse_matrix), show_progress=False,
                               filter_already_liked_items=filter_already_liked, N=num_recommendations)
    recs = pd.DataFrame(recs).reset_index().rename(columns={'index': 'crd_no'})
    recs['crd_no'] = recs['crd_no'].map(lambda x: index_crd_dict[x])
    recs['recs'] = recs.apply(lambda row_df: [int(row_df[x]) for x in range(num_recommendations)], axis=1)
    recs_dict = recs[['crd_no', 'recs']].set_index('crd_no').to_dict()['recs']
    return recs_dict


# метрики на все карты
def fuzzy(user_item_weighted_matrix, model, facts_dict, plu_category_dict, method='mean',
          num_recommendations=5, num_threads=16,
          filter_already_liked=False,
          weight_by_price=True, plu_price=None,
          score_baseline=None):
    """
    Calculates fuzzy precision (cosine_similarity*presence_in_same_category/num_predictions)
    Args:
        user_item_weighted_matrix: pd.core.frame.DataFrame user_item matrix
        model: implicit.als.AlternatingLeastSquares fitted instance or similar
        facts_dict: dictionary of fact sells: {'crd_no': '[plu_1, plu_2, ..., plu_n]'}
        plu_category_dict: dictionary {'plu': 'категория_lvl_2'}
        method: 'mean' or 'max' closeness of recommended item embeddings to true (bought) item embeddings
        num_recommendations: num items to recommend while checking fuzzy
        filter_already_liked: boolean for filtering already presented items
        weight_by_price: boolean. If you need to weight prices.
        plu_price: dict. Dict format {'plu':'price'}
        score_baseline:
        num_threads: Num threads to parallel
    Returns:
        Fuzzy score
    """
    index_crd_dict = user_item_weighted_matrix.reset_index()['index'].to_dict()
    item_factors = pd.DataFrame(model.item_factors, index=user_item_weighted_matrix.columns.tolist())
    user_items_sparse_matrix = sparse.csr_matrix(user_item_weighted_matrix)
    recs_dict = get_recommendations(model, user_items_sparse_matrix, index_crd_dict,
                                    filter_already_liked, num_recommendations)
    id_plu_dict = user_item_weighted_matrix.T.reset_index()['index'].to_dict()
    p = Pool(num_threads)
    facts_split = np.array_split(np.array(list(facts_dict.keys())), num_threads)
    fz_list_of_lists = p.map(partial(_fuzzy_iter,
                                     recs_dict=recs_dict, plu_category_dict=plu_category_dict, facts_dict=facts_dict,
                                     item_factors=item_factors, id_plu_dict=id_plu_dict,
                                     weight_by_price=weight_by_price, plu_price=plu_price,
                                     score_baseline=score_baseline), facts_split)
    if method == 'mean':
        return np.mean([item for sublist in fz_list_of_lists for item in sublist])
    elif method == 'max':
        return np.max([item for sublist in fz_list_of_lists for item in sublist])
    else:
        raise NotImplementedError('Method: %s Not implemented' % method)


def _fuzzy_iter(facts_subset, facts_dict, recs_dict, plu_category_dict,
                item_factors, weight_by_price, id_plu_dict,
                plu_price, score_baseline):
    fz_list = []
    recs_prices_list = []
    for crd in facts_subset:
        recs_list = recs_dict[crd]
        # бейзлайн возвращает сразу plu_id. Поэтому для бейзлайна эта функция не применяется
        if not score_baseline:
            recs_list = [id_plu_dict[x] for x in recs_list]
        if (len(recs_list) > 0) & weight_by_price:
            recs_prices_list = [plu_price[str(rec)] for rec in recs_list]
        facts_list = facts_dict[crd]
        fz = fuzzy_crd(recs_list, facts_list, item_factors,
                       plu_category_dict, weight_by_price, recs_prices_list)
        fz_list.append(fz)
    return fz_list


def precision(user_item_weighted_matrix, model, facts_dict,
              num_recommendations=5,
              filter_already_liked=False, weight_by_price=True, plu_price=None,
              score_baseline=None):
    """
    Calculates fuzzy precision (cosine_similarity*presence_in_same_category/num_predictions)
    Args:
        user_item_weighted_matrix: pd.core.frame.DataFrame user_item matrix
        model: implicit.als.AlternatingLeastSquares fitted instance or similar
        facts_dict: dictionary of fact sells: {'crd_no': '[plu_1, plu_2, ..., plu_n]'}
        num_recommendations: num items to recommend while checking fuzzy
        filter_already_liked: boolean for filtering already presented items
        weight_by_price: boolean. If you need to weight prices.
        plu_price: dict. Dict format {'plu':'price'}
        score_baseline:
    Returns:
        Fuzzy score
    """
    index_crd_dict = user_item_weighted_matrix.reset_index()['index'].to_dict()
    user_items_sparse_matrix = sparse.csr_matrix(user_item_weighted_matrix)
    recs_dict = get_recommendations(model, user_items_sparse_matrix, index_crd_dict,
                                    filter_already_liked, num_recommendations)
    id_plu_dict = user_item_weighted_matrix.T.reset_index()['index'].to_dict()

    recs_prices_list = []
    prec_list = []
    for crd in facts_dict.keys():
        recs_list = recs_dict[crd]
        # бейзлайн возвращает сразу plu_id. Поэтому для бейзлайна эта функция не применяется
        if not score_baseline:
            recs_list = [id_plu_dict[x] for x in recs_list]
        if (len(recs_list) > 0) & weight_by_price:
            recs_prices_list = [plu_price[str(rec)] for rec in recs_list]
        facts_list = facts_dict[crd]
        # Удаляем из словаря фактов для того, чтобы не попадал туда заново
        precision_score = precision_crd(recs_list, facts_list, weight_by_price, recs_prices_list)
        prec_list.append(precision_score)

    return np.mean(prec_list)
