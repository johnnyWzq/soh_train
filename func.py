#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 13:14:29 2019

@author: wuzhiqiang
"""
import sys
lib_dir = "../libs"
sys.path.append(lib_dir)
import g_function as gf
import io_operation as ioo

import re, os

def deal_argv(argv, para_dict):
    """
    第一位代表运行模式，debug/run
    """
    para_len = len(argv)
    if para_len  >= 2:
        import g_function as gf
        para_dict['run_mode'] = gf.deal_1_argv(argv)    
        if para_len > 2:
            print("dealing others parameters...")
            para_dict = deal_other_argv(argv, para_dict)  
    return para_dict

def deal_other_argv(argv, para_dict):
    """
    1:bat_mode; 2:bat_type; 3:bat-structure; 4:bat-year; 5:score_key
    """
    i = 0
    for ar in argv:
        if i == 2: #bat_mode
            regx = '-[0-9a-zA-Z]{3,8}'
            if re.match(regx, ar):
                para_dict['bat_model'] = ar[1:]
                print('The 2nd input parameter is accepted.')
            else:
                print("The 2nd input parameter '%s' is not accepted."%ar)
        elif i == 3: #bat type
            regx = '-[0-9a-zA-Z]{3,6}'
            if re.match(regx, ar):
                para_dict['bat_type'] = ar[1:]
                print('The 3rd input parameter is accepted.')
            else:
                print("The 3rd input parameter '%s' is not accepted."%ar)
        elif i == 4: #bat structrue
            regx = '-[a-zA-Z]{2,6}'
            if re.match(regx, ar):
                para_dict['structure'] = ar[1:]
                print('The 4th input parameter is accepted.')
            else:
                print("The 4thd input parameter '%s' is not accepted."%ar)
        elif i == 5: #bat year
            regx = '-[0-9]{2,4}'
            if re.match(regx, ar):
                para_dict['year'] = ar[1:]
                print('The 5rd input parameter is accepted.')
            else:
                print("The 5rd input parameter '%s' is not accepted."%ar)
        elif i == 6: #score_key
            regx = '-[a-zA-Z]{1,6}'
            if re.match(regx, ar):
                para_dict['score_key'] = ar[1:]
                print('The 5rd input parameter is accepted.')
            else:
                print("The 5rd input parameter '%s' is not accepted."%ar)
        i += 1
    return para_dict

def get_filename_regx(log, **kwds):
    return gf.get_filename_regx(log, **kwds)

def save_workstate_data(data, mask_filename, data_dir):
    tmp = data[data['state'] == 1].reset_index(drop=True) #充电数据
    filename = 'charge_' + mask_filename
    ioo.save_data_csv(tmp, filename, data_dir)
    tmp = data[data['state'] == 2].reset_index(drop=True) #放电数据
    filename = 'discharge_' + mask_filename
    ioo.save_data_csv(tmp, filename, data_dir)
    print('the data has been save within each workstate.')

def get_bat_list(para_dict, mode):
    config = para_dict['config'][mode]
    bat_list = ioo.input_table_name(config)
    print(bat_list)
    return bat_list

def read_bat_data(para_dict, mode, bat_name, limit=None):
    data_dir = para_dict['raw_data_dir'][mode]
    file_name = para_dict['log_raw'] + '_' + bat_name + '.csv'
    raw_data = ioo.read_csv(os.path.join(data_dir, file_name), nrow=limit)
    return raw_data

def save_bat_data(data, filename, para_dict, mode, chunksize=None):
    ioo.save_data_csv(data, filename, para_dict['processed_data_dir'][mode])