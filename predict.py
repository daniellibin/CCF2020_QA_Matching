from tqdm import tqdm, trange
import numpy as np
import pandas as pd
import logging
import torch
import random
import os
from torch import nn, optim
from transformers import BertTokenizer, AdamW, BertModel, BertPreTrainedModel, BertConfig, \
    get_linear_schedule_with_warmup, XLNetModel, XLNetTokenizer, XLNetConfig
from transformers.optimization import get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score
from model import *
from utils import *

import logging
logging.basicConfig(level=logging.DEBUG, filename="train.log",filemode='a')


os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

MODEL_CLASSES = {
   'BertForClass':  BertForClass,
    'BertLastFour':  BertLastFour,
   'BertLastFour_1':  BertLastFour_1,
   'BertDyn':  BertDyn,
    'BertRNN':BertRNN,
    'BertRCNN':BertRCNN
}

class Config:
    def __init__(self):
        # 预训练模型路径
        self.modelId = 0
        self.model = "BertForClass"
        self.Stratification = False
        # self.model_path = '../chinese_wwm_pytorch/'
        # self.model_path = '../chinese_xlnet_mid_pytorch/'
        self.model_path = "../chinese_roberta_wwm_large_ext_pytorch/"
        # self.model_path = "../MC-BERT/"
        # self.model_path = "../ernie/"
        self.num_class = 2
        self.dropout = 0.2
        self.MAX_LEN = 128
        self.epoch = 5
        self.learn_rate = 2e-5
        self.normal_lr = 1e-4
        self.batch_size = 256
        self.k_fold = 5
        self.seed = 2020
        self.device = torch.device('cuda')

        self.pgd = False


config = Config()

random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)


file_path = './log/'
# 创建一个logger
logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)


#semi = pd.read_csv('data/Semi-supervised_test.csv')
#train = pd.concat([train, semi], sort=False)[:20]


train_left = pd.read_csv('./data/train/train.query.tsv',sep='\t',header=None)
train_left.columns=['id','q1']
train_right = pd.read_csv('./data/train/train.reply.tsv',sep='\t',header=None)
train_right.columns=['id','id_sub','q2','label']
train = train_left.merge(train_right, how='left')
train['q2'] = train['q2'].fillna('好的')
test_left = pd.read_csv('./data/test/test.query.tsv',sep='\t',header=None, encoding='gbk')
test_left.columns = ['id','q1']
test_right =  pd.read_csv('./data/test/test.reply.tsv',sep='\t',header=None, encoding='gbk')
test_right.columns=['id','id_sub','q2']
test = test_left.merge(test_right, how='left')


train_query1 = train['q1'].values.astype(str)
train_query2 = train['q2'].values.astype(str)
train_label = train['label'].values.astype(int)

test_query1 = test['q1'].values
test_query2 = test['q2'].values
test_label = [-1] * len(test)

# train.to_csv("train.csv",index=False)
# test.to_csv("test.csv",index=False)

oof_train = np.zeros((len(train), config.num_class), dtype=np.float32)
oof_test = np.zeros((len(test), config.num_class), dtype=np.float32)



kf = KFold(n_splits=config.k_fold, shuffle=True, random_state=config.seed)

for fold, (train_index, valid_index) in enumerate(kf.split(train_query1, train_label)):
    # if fold <= 3:
    #   continue


    model = MODEL_CLASSES[config.model](config).to(config.device)

    PATH = './models/model{}/bert_{}.pth'.format(config.modelId, fold)
    save_model_path = './models/model{}'.format(config.modelId)
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)

    test_D = data_generator([test_query1, test_query2, test_label], config)
    model = torch.load(PATH).to(config.device)
    model.eval()
    with torch.no_grad():
        res = []
        pred_logit = None

        for input_ids, input_masks, segment_ids, labels in tqdm(test_D):
            input_ids = torch.tensor(input_ids).to(config.device)
            input_masks = torch.tensor(input_masks).to(config.device)
            segment_ids = torch.tensor(segment_ids).to(config.device)

            y_pred = model(input_ids, input_masks, segment_ids)
            y_pred = y_pred.detach().to("cpu")

            if pred_logit is None:
                pred_logit = y_pred
            else:
                pred_logit = np.vstack((pred_logit, y_pred))


    oof_test += np.array(pred_logit)


    del model
    torch.cuda.empty_cache()

oof_test /= config.k_fold
save_result_path = './result'
if not os.path.exists(save_result_path):
    os.makedirs(save_result_path)


test['label'] = np.argmax(oof_test,axis=1)
test[['id','id_sub','label']].to_csv("./result/result{}_修正前.csv".format(config.modelId), index=False, header=None, sep='\t')


from functools import partial
import numpy as np
import scipy as sp
from sklearn.metrics import f1_score
class OptimizedF1(object):
    def __init__(self):
        self.coef_ = []

    def _kappa_loss(self, coef, X, y):
        """
        y_hat = argmax(coef*X, axis=-1)
        :param coef: (1D array) weights
        :param X: (2D array)logits
        :param y: (1D array) label
        :return: -f1
        """
        X_p = np.copy(X)
        X_p = coef*X_p
        # ll = f1_score(y, np.argmax(X_p, axis=-1), average='macro')
        ll = f1_match(y, np.argmax(X_p, axis=-1))
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [1. for _ in range(len(set(y)))]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, y):
        X_p = np.copy(X)
        X_p = self.coef_['x'] * X_p
        # return f1_score(y, np.argmax(X_p, axis=-1), average='macro')
        return f1_match(y, np.argmax(X_p, axis=-1))

    def coefficients(self):
        return self.coef_['x']

op = OptimizedF1()
op.fit(oof_train,train_label)

oof_test_optimizer = op.coefficients()*oof_test

test['label'] = np.argmax(oof_test_optimizer,axis=1)


test[['id','id_sub','label']].to_csv("./result/result{}_修正后.csv".format(config.modelId), index=False, header=None, sep='\t')

np.save('./result/oof_train{}.npy'.format(config.modelId), oof_train)
np.save('./result/oof_test{}.npy'.format(config.modelId), oof_test)
np.save('./result/oof_test_optimizer{}.npy'.format(config.modelId), oof_test_optimizer)