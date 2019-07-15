#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 14:09:24 2019

@author: wuzhiqiang
"""

print(__doc__)

import sys
import os, re
import func as fc
import pandas as pd

app_dir = "soh_train"


def init_data_para():
    para_dict = {}
    para_dict['raw_data_dir'] = {'debug': os.path.normpath('./data/raw'),
                                 'run': os.path.normpath('/raid/data/raw/' + app_dir)}
    para_dict['processed_data_dir'] = {'debug': os.path.normpath('./data/processed'),
                                        'run': os.path.normpath('/raid/data/processed_data/complete/' + app_dir)}
    para_dict['config'] = {'debug': {'s': '192.168.1.105', 'u': 'data', 'p': 'For2019&tomorrow', 'db': 'test_bat', 'port': 3306},
                             'run': {'s': 'localhost', 'u': 'data', 'p': 'For2019&tomorrow', 'db': 'test_bat', 'port': 3306}
                             #'run': {'s': 'localhost', 'u': 'root', 'p': 'wzqsql', 'db': 'cell_lg36', 'port': 3306}
                             }
    para_dict['data_limit'] = {'debug': 1000,
                                 'run': None}
    
    para_dict['score_key'] = 'soh'
    para_dict['cell_key'] = 'cell_name'
    para_dict['states'] = ['charge', 'discharge']
    para_dict['log_raw'] = 'cell_soh'
    para_dict['log_pro'] = 'processed'
    
    para_dict['run_mode'] = 'debug'
    return para_dict

def main(argv):
    
    para_dict =  init_data_para()
    para_dict = fc.deal_argv(argv, para_dict)
    mode = para_dict['run_mode']

    #读取所需的数据，并进行处理后存储到指定位置
    print('starting processing the data...')
    bat_list = fc.get_bat_list(para_dict, mode)
    regx, mask_filename = fc.get_filename_regx(para_dict['log_pro'], **para_dict)
    if bat_list is not None:
        process_data = []
        for bat_name in bat_list:
            one_bat_data = fc.read_bat_data(para_dict, mode, bat_name, limit=para_dict['data_limit'][mode])
            one_bat_data[para_dict['cell_key']] = bat_name
            process_data.append(one_bat_data)
        process_data = pd.concat(process_data)
        fc.save_bat_data(process_data, mask_filename, para_dict, mode)#存储处理后的数据
    else:
         print('there is no bat!')
    
    #将处理好的数据按工作状态划分后存储到指定位置
    print('save the processed data...')
    fc.save_workstate_data(process_data, mask_filename, para_dict['processed_data_dir'][mode])
    
    #训练模型
    print('starting training the model...')
    for state in para_dict['states']:
        file_name = r'%s_%s'%(state, mask_filename)
        para_dict['pkl_dir'] = {'run': os.path.normpath('/raid/data/processed_data/pkl/'+ app_dir +'/%s_pkl'%state),
                                 'debug': os.path.normpath('./data/%s_pkl'%state)}
        import build_model as bm
        bm.train_model(file_name, state, **para_dict)
    
if __name__ == '__main__':
    main(sys.argv)