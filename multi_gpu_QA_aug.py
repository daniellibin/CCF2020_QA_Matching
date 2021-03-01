import argparse

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
    'BertForClass': BertForClass,
    'BertForClass_MultiDropout': BertForClass_MultiDropout,
    'BertLastTwoCls': BertLastTwoCls,
    'BertLastTwoClsPooler': BertLastTwoClsPooler,
    'BertLastTwoEmbeddings': BertLastTwoEmbeddings,
    'BertLastTwoEmbeddingsPooler': BertLastTwoEmbeddingsPooler,
    'BertLastFourCls': BertLastFourCls,
    'BertLastFourClsPooler': BertLastFourClsPooler,
    'BertLastFourEmbeddings': BertLastFourEmbeddings,
    'BertLastFourEmbeddingsPooler': BertLastFourEmbeddingsPooler,
    'BertDynCls': BertDynCls,
    'BertDynEmbeddings': BertDynEmbeddings,
    'BertRNN': BertRNN,
    'BertCNN': BertCNN,
    'BertRCNN': BertRCNN,
    'XLNet': XLNet,
    'Electra': Electra

}
parser = argparse.ArgumentParser()
parser.add_argument('--modelId', default='0', type=int)
parser.add_argument(
        "--model",
        default='BertLastFourCls',
        type=str,
    )
parser.add_argument(
        "--Stratification",
        default=False,
        type=bool,
    )
parser.add_argument(
        "--model_path",
        default='../RoBERTa-large-pair/',
        type=str,
    )
parser.add_argument(
        "--dropout",
        default=0,
        type=float,
    )
parser.add_argument(
        "--MAX_LEN",
        default=128,
        type=int,
    )
parser.add_argument(
        "--epoch",
        default=5,
        type=int,
    )
parser.add_argument(
        "--learn_rate",
        default=2e-5,
        type=float,
    )

parser.add_argument(
        "--batch_size",
        default=128,
        type=int,
    )
parser.add_argument(
        "--k_fold",
        default=5,
        type=int,
    )
parser.add_argument(
        "--seed",
        default=42,
        type=int,
    )
parser.add_argument(
        "--focalloss",
        default=False,
        type=bool,
    )
parser.add_argument(
        "--pgd",
        default=True,
        type=bool,
    )
parser.add_argument(
        "--fgm",
        default=False,
        type=bool,
    )

args = parser.parse_args()

class Config:
    def __init__(self):
        # 预训练模型路径
        self.modelId = args.modelId
        self.model = args.model
        self.Stratification = args.Stratification
        # self.model_path = 'D:/Pretrain_model/chinese-electra-180g-large-discriminator/'
        # self.model_path = '../chinese_wwm_pytorch/'
        # self.model_path = '../chinese_wwm_ext_pytorch/'
        # self.model_path = "../chinese_roberta_wwm_ext_pytorch/"
        # self.model_path = "../chinese_roberta_wwm_large_ext_pytorch/"
        # self.model_path = '../chinese_xlnet_base_pytorch/'
        # self.model_path = '../chinese_xlnet_mid_pytorch/'

        # self.model_path = '../chinese-electra-180g-large-discriminator/'
        # self.model_path = '../ernie-1.0/'
        # self.model_path = '../RoBERTa-large-pair/'
        self.model_path = args.model_path
        self.num_class = 2
        self.dropout = args.dropout
        self.MAX_LEN = args.MAX_LEN
        self.epoch = args.epoch
        self.learn_rate = args.learn_rate
        self.normal_lr = 1e-4
        self.batch_size = args.batch_size
        self.k_fold = args.k_fold
        self.seed = args.seed

        self.device = torch.device('cuda')
        # self.device = torch.device('cpu')

        self.focalloss = args.focalloss
        self.pgd = args.pgd
        self.fgm = args.fgm


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
#train_left = pd.read_csv('./data/train/train.query.eda.tsv',sep=',',header=None)
train_left.columns=['id','q1']
train_right = pd.read_csv('./data/train/train.reply.tsv',sep='\t',header=None)
#train_right = pd.read_csv('./data/train/train.reply.eda.tsv',sep=',',header=None)
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
    print('\n\n------------fold:{}------------\n'.format(fold))

    q1 = train_query1[train_index]
    q2 = train_query2[train_index]
    y = train_label[train_index]

    val_q1 = train_query1[valid_index]
    val_q2 = train_query2[valid_index]
    val_y = train_label[valid_index]

    train_D = data_generator([q1, q2, y], config, shuffle=True)
    val_D = data_generator([val_q1, val_q2, val_y], config)

    model = MODEL_CLASSES[config.model](config).to(config.device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)


    if config.pgd:
        pgd = PGD(model)
        K = 3

    elif config.fgm:
        fgm = FGM(model)

    if config.focalloss:
        loss_fn = FocalLoss(config.num_class)
    else:
        loss_fn = nn.CrossEntropyLoss()  # BCEWithLogitsLoss就是把Sigmoid-BCELoss合成一步


    num_train_steps = int(len(train) / config.batch_size * config.epoch)
    param_optimizer = list(model.named_parameters())

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    if config.Stratification:
        bert_params = [x for x in param_optimizer if 'bert' in x[0]]
        normal_params = [p for n, p in param_optimizer if 'bert' not in n]
        optimizer_parameters = [
            {'params': [p for n, p in bert_params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in bert_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': normal_params, 'lr': config.normal_lr},
        ]
    else:
        optimizer_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]

    optimizer = AdamW(optimizer_parameters, lr=config.learn_rate) # lr为全局学习率
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(len(train) / config.batch_size / 2),
        num_training_steps=num_train_steps
    )

    best_f1 = 0
    PATH = './models/model{}/bert_{}.pth'.format(config.modelId, fold)
    save_model_path = './models/model{}'.format(config.modelId)
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)

    for e in range(config.epoch):
        print('\n------------epoch:{}------------'.format(e))
        model.train()
        acc = 0
        train_len = 0
        loss_num = 0
        tq = tqdm(train_D)

        for input_ids, input_masks, segment_ids, labels in tq:
            input_ids = torch.tensor(input_ids).to(config.device)
            input_masks = torch.tensor(input_masks).to(config.device)
            segment_ids = torch.tensor(segment_ids).to(config.device)
            label_t = torch.tensor(labels, dtype=torch.long).to(config.device)

            y_pred = model(input_ids, input_masks, segment_ids)

            loss = loss_fn(y_pred, label_t)
            loss = loss.mean()
            loss.backward()

            if config.pgd:
                pgd.backup_grad()
                # 对抗训练
                for t in range(K):
                    pgd.attack(is_first_attack=(t == 0))  # 在embedding上添加对抗扰动, first attack时备份param.data
                    if t != K - 1:
                        model.zero_grad()
                    else:
                        pgd.restore_grad()
                    y_pred = model(input_ids, input_masks, segment_ids)

                    loss_adv = loss_fn(y_pred, label_t)
                    loss_adv = loss_adv.mean()
                    loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                pgd.restore()  # 恢复embedding参数

            elif config.fgm:
                # 对抗训练
                fgm.attack()  # 在embedding上添加对抗扰动
                y_pred = model(input_ids, input_masks, segment_ids)
                loss_adv = loss_fn(y_pred, label_t)
                loss_adv = loss_adv.mean()
                loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                fgm.restore()  # 恢复embedding参数


            # 梯度下降，更新参数
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()

            y_pred = np.argmax(y_pred.detach().to("cpu").numpy(), axis=1)
            acc += sum(y_pred == labels)
            loss_num += loss.item()
            train_len += len(labels)
            tq.set_postfix(fold=fold, epoch=e, loss=loss_num / train_len, acc=acc / train_len)

        model.eval()
        with torch.no_grad():
            y_p = []
            train_logit = None
            for input_ids, input_masks, segment_ids, labels in tqdm(val_D):
                input_ids = torch.tensor(input_ids).to(config.device)
                input_masks = torch.tensor(input_masks).to(config.device)
                segment_ids = torch.tensor(segment_ids).to(config.device)
                label_t = torch.tensor(labels, dtype=torch.long).to(config.device)

                y_pred = model(input_ids, input_masks, segment_ids)
                y_pred = y_pred.detach().to("cpu").numpy()
                if train_logit is None:
                    train_logit = y_pred
                else:
                    train_logit = np.vstack((train_logit, y_pred))

                y_pred = np.argmax(y_pred, axis=1)
                y_p += list(y_pred)

            f1 = f1_score(val_y, y_p, average="macro")
            f1_qa = f1_match(val_y, y_p)
            print("best_f1:{}  f1_qa:{}  f1:{}\n".format(best_f1, f1_qa, f1))
            if f1_qa >= best_f1:
                best_f1 = f1_qa
                oof_train[valid_index] = np.array(train_logit)
                torch.save(model.module if hasattr(model, "module") else model, PATH)

    test_D = data_generator([test_query1, test_query2, test_label], config)
    model = torch.load(PATH).to(config.device)
    model.eval()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
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

    optimizer.zero_grad()

    del model
    torch.cuda.empty_cache()

oof_test /= config.k_fold
save_result_path = './result'
if not os.path.exists(save_result_path):
    os.makedirs(save_result_path)


test['label'] = np.argmax(oof_test,axis=1)
test[['id','id_sub','label']].to_csv("./result/result{}_original.csv".format(config.modelId), index=False, header=None, sep='\t')


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


test[['id','id_sub','label']].to_csv("./result/result{}_optimizer.csv".format(config.modelId), index=False, header=None, sep='\t')

np.save('./result/oof_train{}.npy'.format(config.modelId), oof_train)
np.save('./result/oof_test{}.npy'.format(config.modelId), oof_test)
np.save('./result/oof_test_optimizer{}.npy'.format(config.modelId), oof_test_optimizer)