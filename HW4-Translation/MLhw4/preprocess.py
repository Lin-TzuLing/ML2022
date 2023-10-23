import re
import pandas as pd
import os
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("dict_path",type=str, help="dictionary path")
parser.add_argument("ch_path",type=str, help="training data path")
parser.add_argument("tl_path",type=str, help="training data label path")
parser.add_argument("test_path",type=str, help="testing data path")
parser.add_argument("result_path",type=str, help="store preprocessed training data/label")
parser.add_argument("test_result_path",type=str, help="store preprocessed testing data")
args = parser.parse_args()

# path
dict_path = args.dict_path
if not os.path.exists(dict_path):
    os.makedirs(dict_path)
ch_path = args.ch_path
tl_path = args.tl_path
test_path = args.test_path
result_path = args.result_path
test_result_path = args.test_result_path

# read data
ch_df = pd.read_csv(ch_path, names=['id','raw_ch'], header=None, skiprows = 1).applymap(str)
tl_df = pd.read_csv(tl_path, names=['id','raw_tl'], header=None, skiprows = 1).applymap(str)
test_ch_df = pd.read_csv(test_path, names=['id','raw_ch'], header=None, skiprows = 1).applymap(str)

def get_word_list(s1):
    res = re.compile(r"([\u3002\uff1b\uff0c\uff1a\u201c\u201d\uff08\uff09\u3001\uff1f\u300a\u300b\u4e00-\u9fa5])")
    p1 = res.split(s1)
    str1_list = []
    for str in p1:
        if res.split(str) == None:
            str1_list.append(str)
        else:
            ret = res.split(str)
            for ch in ret:
                str1_list.append(ch)
    # no space (match test data)
    list_word1 = [w for w in str1_list if len(w.strip()) > 0]
    return  list_word1

def get_word_list_target(s1):
    p1 = re.split(r'(\s+|\-+)', s1)
    str1_list = []
    for str in p1:
        if len(str)>0:
            str1_list.append(str)
    list_word1 = [w for w in str1_list if len(w) > 0]
    return list_word1


# process training data (source language, CH)
ch_df['ch'] = ''
ch_dict = {'tmp':-1}
for i in range(len(ch_df)):
    txt_new = ch_df.iloc[i]['raw_ch']
    txt_new = get_word_list(txt_new)
    for word in txt_new:
        if word not in ch_dict.keys() :
            ch_dict[word]=max(ch_dict.values())+1
    ch_df.at[i, 'ch'] = txt_new
del ch_dict['tmp']


# process training data (target language, TL)
tl_df['tl'] = ''
tl_dict = {'tmp':-1}
for i in range(len(tl_df)):
    txt_new = tl_df.iloc[i]['raw_tl']
    txt_new = get_word_list_target(txt_new)
    for word in txt_new:
        if word not in tl_dict.keys() :
            tl_dict[word]=max(tl_dict.values())+1
    tl_df.at[i, 'tl'] = txt_new
del tl_dict['tmp']

# prepare test CH data
test_ch_df['ch'] = ''
for i in range(len(test_ch_df)):
    txt_new = test_ch_df.iloc[i]['raw_ch']
    txt_new = get_word_list(txt_new)
    test_ch_df.at[i, 'ch'] = txt_new

# update dict
with open(dict_path+'ch.pkl', 'wb') as file:
    pickle.dump(ch_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
with open(dict_path+'tl.pkl', 'wb') as file:
    pickle.dump(tl_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

# save processed training data and testing data
df = pd.concat([ch_df.set_index('id'),tl_df.set_index('id')], axis=1, join='inner')
df.to_pickle(result_path, compression='gzip')
test_ch_df.to_pickle(test_result_path, compression='gzip')