# -*- coding: utf-8 -*-
# @Time    : 2018/10/15 3:23 PM
# @Author  : Inf.Turing
# @Site    :
# @File    : lgb_baseline.py
# @Software: PyCharm

# 不要浪费太多时间在自己熟悉的地方，要学会适当的绕过一些
# 良好的阶段性收获是坚持的重要动力之一
# 用心做事情，一定会有回报
import datetime
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
import lightgbm as lgb
import time
import pandas as pd
import numpy as np

path = '../data'
# 全量数据
data = pd.read_csv(path + '/data_all.csv')
data = data.fillna(-1)
data['day'] = data['time'].apply(lambda x: int(time.strftime("%d", time.localtime(x))))
data['hour'] = data['time'].apply(lambda x: int(time.strftime("%H", time.localtime(x))))
data['label'] = data.click.astype(int)
del data['click']
bool_feature = ['creative_is_jump', 'creative_is_download', 'creative_is_js', 'creative_is_voicead',
                'creative_has_deeplink', 'app_paid']
for i in bool_feature:
    data[i] = data[i].astype(int)
data['advert_industry_inner_1'] = data['advert_industry_inner'].apply(lambda x: x.split('_')[0])
ad_cate_feature = ['adid', 'advert_id', 'orderid', 'advert_industry_inner_1', 'advert_industry_inner', 'advert_name',
                   'campaign_id', 'creative_id', 'creative_type', 'creative_tp_dnf', 'creative_has_deeplink',
                   'creative_is_jump', 'creative_is_download']
media_cate_feature = ['app_cate_id', 'f_channel', 'app_id', 'inner_slot_id']
content_cate_feature = ['city', 'carrier', 'province', 'nnt', 'devtype', 'osv', 'os', 'make', 'model']
origin_cate_list = ad_cate_feature + media_cate_feature + content_cate_feature
# 编码，加速
for i in origin_cate_list:
    data[i] = data[i].map(
        dict(zip(data[i].unique(), range(0, data[i].nunique()))))
count_feature_list = []


def feature_count(data, features=[], is_feature=True):
    if len(set(features)) != len(features):
        print('equal feature !!!!')
        return data
    new_feature = 'count'
    nunique = []
    for i in features:
        nunique.append(data[i].nunique())
        new_feature += '_' + i.replace('add_', '')
    if len(features) > 1 and len(data[features].drop_duplicates()) <= np.max(nunique):
        print(new_feature, 'is unvalid cross feature:')
        return data
    temp = data.groupby(features).size().reset_index().rename(columns={0: new_feature})
    data = data.merge(temp, 'left', on=features)
    if is_feature:
        count_feature_list.append(new_feature)
    if 'day_' in new_feature:
        print('fix:', new_feature)
        data.loc[data.day == 3, new_feature] = data[data.day == 3][new_feature] * 4
    return data


for i in origin_cate_list:
    n = data[i].nunique()
    if n > 5:
        data = feature_count(data, [i])
        data = feature_count(data, ['day', 'hour', i])

ratio_feature_list = []
for i in ['adid']:
    for j in content_cate_feature:
        data = feature_count(data, [i, j])
        if data[i].nunique() > 5 and data[j].nunique() > 5:
            data['ratio_' + j + '_of_' + i] = data[
                                                  'count_' + i + '_' + j] / data['count_' + i]
            data['ratio_' + i + '_of_' + j] = data[
                                                  'count_' + i + '_' + j] / data['count_' + j]
            ratio_feature_list.append('ratio_' + j + '_of_' + i)
            ratio_feature_list.append('ratio_' + i + '_of_' + j)

for i in media_cate_feature:
    for j in content_cate_feature + ad_cate_feature:
        new_feature = 'inf_' + i + '_' + j
        data = feature_count(data, [i, j])
        if data[i].nunique() > 5 and data[j].nunique() > 5:
            data['ratio_' + j + '_of_' + i] = data[
                                                  'count_' + i + '_' + j] / data['count_' + i]
            data['ratio_' + i + '_of_' + j] = data[
                                                  'count_' + i + '_' + j] / data['count_' + j]
            ratio_feature_list.append('ratio_' + j + '_of_' + i)
            ratio_feature_list.append('ratio_' + i + '_of_' + j)

cate_feature = origin_cate_list
num_feature = ['creative_width', 'creative_height', 'hour'] + count_feature_list + ratio_feature_list
feature = cate_feature + num_feature
print(len(feature), feature)
# 低频过滤
for feature in cate_feature:
    if 'count_' + feature in data.keys():
        print(feature)
        data.loc[data['count_' + feature] < 2, feature] = -1
        data[feature] = data[feature] + 1
predict = data[(data.label == -1) & (data.data_type == 2)]
predict_result = predict[['instance_id']]
predict_result['predicted_score'] = 0
predict_x = predict.drop('label', axis=1)
train_x = data[data.label != -1].reset_index(drop=True)
train_y = train_x.pop('label').values
base_train_csr = sparse.csr_matrix((len(train_x), 0))
base_predict_csr = sparse.csr_matrix((len(predict_x), 0))

enc = OneHotEncoder()
for feature in cate_feature:
    enc.fit(data[feature].values.reshape(-1, 1))
    base_train_csr = sparse.hstack((base_train_csr, enc.transform(train_x[feature].values.reshape(-1, 1))), 'csr',
                                   'bool')
    base_predict_csr = sparse.hstack((base_predict_csr, enc.transform(predict[feature].values.reshape(-1, 1))),
                                     'csr',
                                     'bool')
print('one-hot prepared !')

cv = CountVectorizer(min_df=20)
for feature in ['user_tags']:
    data[feature] = data[feature].astype(str)
    cv.fit(data[feature])
    base_train_csr = sparse.hstack((base_train_csr, cv.transform(train_x[feature].astype(str))), 'csr', 'bool')
    base_predict_csr = sparse.hstack((base_predict_csr, cv.transform(predict_x[feature].astype(str))), 'csr',
                                     'bool')
print('cv prepared !')

train_csr = sparse.hstack(
    (sparse.csr_matrix(train_x[num_feature]), base_train_csr), 'csr').astype(
    'float32')
predict_csr = sparse.hstack(
    (sparse.csr_matrix(predict_x[num_feature]), base_predict_csr), 'csr').astype('float32')
lgb_model = lgb.LGBMClassifier(
    boosting_type='gbdt', num_leaves=61, reg_alpha=3, reg_lambda=1,
    max_depth=-1, n_estimators=5000, objective='binary',
    subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
    learning_rate=0.035, random_state=2018, n_jobs=10
)
skf = StratifiedKFold(n_splits=5, random_state=2018, shuffle=True)
best_score = []
for index, (train_index, test_index) in enumerate(skf.split(train_csr, train_y)):
    lgb_model.fit(train_csr[train_index], train_y[train_index],
                  eval_set=[(train_csr[train_index], train_y[train_index]),
                            (train_csr[test_index], train_y[test_index])], early_stopping_rounds=200, verbose=10)
    best_score.append(lgb_model.best_score_['valid_1']['binary_logloss'])
    print(best_score)
    test_pred = lgb_model.predict_proba(predict_csr, num_iteration=lgb_model.best_iteration_)[:, 1]
    predict_result['predicted_score'] = predict_result['predicted_score'] + test_pred
predict_result['predicted_score'] = predict_result['predicted_score'] / 5
mean = predict_result['predicted_score'].mean()
print('mean:', mean)
now = datetime.datetime.now()
now = now.strftime('%m-%d-%H-%M')
predict_result[['instance_id', 'predicted_score']].to_csv(path + "/submission/lgb_baseline_%s.csv" % now, index=False)
