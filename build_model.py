#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 16:40:38 2019

@author: wuzhiqiang
"""


print(__doc__)

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
    
import sys, os
lib_dir = "../libs"
sys.path.append(lib_dir)
import utils as ut
import io_operation as ioo
import feature_function as ff

def sel_feature(data, *kwg):
    print('selecting the feature...')
    col_names = []
    for i in data.columns:
        for j in kwg:
            if j in i:
                col_names.append(i)
                break
    for i in col_names:
        tmp = data[i]
        data['feature_' + i] = tmp
    print('Done.')
    return data

def calc_feature_data(config, cell_key, score_key, file_dir, file_name, data=None):
    """
    """
    if data == None:
        data = ioo.read_csv(os.path.join(file_dir, file_name + '.csv'))
    feature_name = ['dqdv', 'voltage']
    data = sel_feature(data, *feature_name)
    if data is None:
        return None, None, None
    data = ff.calc_score(data, score_key)
    data = data[data['score'] > 0.1]
    
    print('getting the feature...')
    data_x = data[[i for i in data.columns if 'feature_' in i]]
    #将电流特征去掉
    #data_x = data_x.drop([i for i in data.columns if '_current_' in i], axis=1)
    data_x = ff.drop_feature(data_x)
    data_y = data['score']
    
    cell_key = data[[cell_key]]
    #cell_no = cell_no.reset_index(drop=True)
    
    return data_x, data_y, cell_key

def build_model(data_x, data_y, split_mode='test',
                feature_method='f_regression', feature_num=100, pkl_dir='pkl'):
    # select features
    # feature_num: integer that >=0
    # method: ['f_regression', 'mutual_info_regression', 'pca']
    data_x, min_max = ut.select_feature(data_x, data_y, method=feature_method, feature_num=feature_num)
    #if min_max is not None:
       # min_max.to_csv(os.path.join(pkl_dir, 'min_max.csv'), index=False, index_label=False, encoding='gb18030')
    # start building model
    np_x = np.nan_to_num(data_x.values)
    np_y = np.nan_to_num(data_y.values)
    print('train_set.shape=%s, test_set.shape=%s' %(np_x.shape, np_y.shape))

    res = {}
    if split_mode == 'test':
        x_train, x_val, y_train, y_val = train_test_split(data_x, data_y, test_size=0.2,
                                                          shuffle=True)
        model = LinearRegression()
        res['lr'] = ut.test_model(model, x_train, x_val, y_train, y_val)
        ut.save_model(model, data_x.columns, pkl_dir)
        model = DecisionTreeRegressor()
        res['dt'] = ut.test_model(model, x_train, x_val, y_train, y_val)
        ut.save_model(model, data_x.columns, pkl_dir, depth=5)
        model = RandomForestRegressor()
        res['rf'] = ut.test_model(model, x_train, x_val, y_train, y_val)
        ut.save_model(model, data_x.columns, pkl_dir, depth=5)
        model = GradientBoostingRegressor()
        res['gbdt'] = ut.test_model(model, x_train, x_val, y_train, y_val)
        ut.save_model(model, data_x.columns, pkl_dir)
    elif split_mode == 'cv':
        model = LinearRegression()
        res['lr'] = ut.cv_model(model, np_x, np_y)
        model = DecisionTreeRegressor()
        res['dt'] = ut.cv_model(model, np_x, np_y)
        model = RandomForestRegressor()
        res['rf'] = ut.cv_model(model, np_x, np_y)
        model = GradientBoostingRegressor()
        res['gbdt'] = ut.cv_model(model, np_x, np_y)
    else:
        print('parameter mode=%s unresolved' % (model))
        
    return res
    
def train_model(file_name, state, **kwds):
    for key, value in kwds.items():
        if key == 'cell_key':
            cell_key = value
        elif key == 'state':
            state = value
        elif key == 'pkl_dir':
            pkl_dir = value[kwds['run_mode']]
        elif key == 'config':
            config = value[kwds['run_mode']]
        elif key == 'processed_data_dir':
            file_dir = value[kwds['run_mode']]
        elif key == 'score_key':
            score_key = value
    data_x, data_y, cell_key = calc_feature_data(config, cell_key, score_key, file_dir, file_name)
    if data_x is None:
        print('there is no data to train.')
        return
    ioo.save_data_csv(data_x.head(), 'columns_feature_'+file_name, pkl_dir)
    mode = 'test'
    res = build_model(data_x, data_y, split_mode=mode, feature_method='f_regression', pkl_dir=pkl_dir)
    if mode == 'test':
        d = {'lr':'线性回归(LR)', 'dt':'决策树回归', 'rf':'随机森林', 'gbdt':'GBDT',
            'eva':'评估结果'}
        ioo.save_model_result(res, d, pkl_dir, state)