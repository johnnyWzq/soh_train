#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 11:47:59 2019

@author: wuzhiqiang

#通用的soh训练框架
"""

print(__doc__)

import sys
import os, re

app_dir = "../zc"
m = re.match(r'(\.\.\/)(\w+)', app_dir)
save_dir = m.group(2)
#lib_dir = "../libs"

def init_data_para():
    para_dict = {}
    para_dict['raw_data_dir'] = {'debug': os.path.normpath('./data/processed_data/raw'),
                                 'run': os.path.normpath('/raid/data/raw/' + save_dir)}
    para_dict['processed_data_dir'] = {'debug': os.path.normpath(app_dir + '/data'),
                                        'run': os.path.normpath('/raid/data/processed_data/complete/' + save_dir)}
    para_dict['scale_data_dir'] = {'debug': os.path.normpath(app_dir + '/data'),
                                    'run': os.path.normpath('/raid/data/processed_data/scale/' + save_dir)}
    para_dict['config'] = {'debug': {'s': '192.168.1.105', 'u': 'data', 'p': 'For2019&tomorrow', 'db': 'test_bat', 'port': 3306},
                             'run': {'s': 'localhost', 'u': 'data', 'p': 'For2019&tomorrow', 'db': 'test_bat', 'port': 3306}
                             #'run': {'s': 'localhost', 'u': 'root', 'p': 'wzqsql', 'db': 'cell_lg36', 'port': 3306}
                             }
    para_dict['data_limit'] = {'debug': 1000,
                                 'run': None}
    
    para_dict['score_key'] = 'c'
    para_dict['cell_key'] = 'cell_no'
    para_dict['states'] = ['charge', 'discharge']
    para_dict['log_pro'] = 'processed'
    para_dict['log_scale'] = 'scale'
    
    para_dict['start_kwd'] = 'start_tick'
    para_dict['end_kwd'] = 'end_tick'
    
    para_dict['run_mode'] = 'debug'
    return para_dict

def main(argv):
    sys.path.append(app_dir)
    import rw_bat_data as rwd
    import preprocess_data as ppd
    import func as fc
    import scale_data as sd
    
    para_dict =  init_data_para()
    para_dict = fc.deal_argv(argv, para_dict)
    mode = para_dict['run_mode']

    #读取所需的数据，并进行处理后存储到指定位置
    print('starting processing the data...')
    bat_list = rwd.get_bat_list(para_dict, mode)
    regx, mask_filename = fc.get_filename_regx(para_dict['log_pro'], **para_dict)
    
    if bat_list is not None:
        for bat_name in bat_list:
            raw_data = rwd.read_bat_data(para_dict, mode, bat_name, limit=para_dict['data_limit'][mode])
            data = ppd.preprocess_data(bat_name, raw_data)
            rwd.save_bat_data(data, para_dict['log_pro']+'_'+bat_name, para_dict, mode)#存储处理后的数据
    else:
         print('there is no bat!')
    
    #将处理好的数据按工作状态划分后存储到指定位置
    print('save the processed data...')
    
    result = fc.save_workstate_data(regx, mask_filename, para_dict['processed_data_dir'][mode], para_dict['processed_data_dir'][mode])
    if not result:
        print('there is not any files included the data which would been scaled.')
        return
    else:
        #进行扩充
        print('to be scaled...')
        
        for state in para_dict['states']:
            file_name = r'%s_'%state + mask_filename
            processed_data = sd.get_processed_data(os.path.join(para_dict['processed_data_dir'][mode], file_name))
            scale_data = sd.generate_data(processed_data, **para_dict)
            sd.save_scale_data(scale_data, para_dict['log_scale']+'_'+file_name, para_dict['scale_data_dir'][mode])
            print('finished scaling the %s data'%state)
    
    #训练模型
    print('starting training the model...')
    for state in para_dict['states']:
        file_name = r'%s_%s_%s'%(para_dict['log_scale'], state, mask_filename)
        para_dict['pkl_dir'] = {'run': os.path.normpath('/raid/data/processed_data/pkl/'+save_dir+'/%s_pkl'%state),
                                 'debug': os.path.normpath(app_dir + '/%s_pkl'%state)}
        import build_model as bm
        bm.train_model(file_name, state, **para_dict)
    
if __name__ == '__main__':
    main(sys.argv)