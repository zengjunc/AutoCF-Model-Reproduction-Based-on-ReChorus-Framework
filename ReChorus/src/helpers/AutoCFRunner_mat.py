import os
import gc
import torch
import torch.nn as nn
import logging
import numpy as np
from time import time
from tqdm import tqdm
from torch.utils.data import DataLoader
import pickle
from utils import utils
from utils.utils import calcRegLoss, contrast
from models.BaseModel import BaseModel
from models.general.AutoCF_mat import AutoCF_mat, RandomMaskSubgraphs, LocalGraph
from helpers.BaseRunner import BaseRunner
from helpers.AutoCFReader_mat import AutoCFReader_mat
from typing import Dict, List

class AutoCFRunner_mat(object):
    @staticmethod
    def parse_runner_args(parser):
        parser.add_argument('--epoch', type=int, default=100,
                            help='Number of epochs.')
        parser.add_argument('--check_epoch', type=int, default=1,
                            help='Check some tensors every check_epoch.')
        parser.add_argument('--test_epoch', type=int, default=-1,
                            help='Print test results every test_epoch (-1 means no print).')
        parser.add_argument('--early_stop', type=int, default=10,
                            help='The number of epochs when dev results drop continuously.')
        parser.add_argument('--lr', type=float, default=1e-3,
                            help='Learning rate.')
        parser.add_argument('--l2', type=float, default=0,
                            help='Weight decay in optimizer.')
        parser.add_argument('--batch_size', type=int, default=4096,
                            help='Batch size during training.')
        parser.add_argument('--eval_batch_size', type=int, default=256,
                            help='Batch size during testing.')
        parser.add_argument('--optimizer', type=str, default='Adam',
                            help='optimizer: SGD, Adam, Adagrad, Adadelta')
        parser.add_argument('--num_workers', type=int, default=0,
                            help='Number of processors when prepare batches in DataLoader')
        parser.add_argument('--pin_memory', type=int, default=0,
                            help='pin_memory in DataLoader')
        parser.add_argument('--topk', type=str, default='5,10,20,50',
                            help='The number of items recommended to each user.')
        parser.add_argument('--metric', type=str, default='NDCG,HR',
                            help='metrics: NDCG, HR')
        parser.add_argument('--main_metric', type=str, default='',
                            help='Main metric to determine the best model.')
        parser.add_argument('--fixSteps', default=10, type=int, help='steps to train on the same sampled graph')
        parser.add_argument('--ssl_reg', default=1, type=float, help='contrastive regularizer')
        parser.add_argument('--tstBat', default=256, type=int, help='number of users in a testing batch')
        parser.add_argument('--save_path', default='tem', help='file name to save model and training record')
        return parser

    # 静态方法，评估参数
    # @staticmethod
    # def evaluate_method(predictions: np.ndarray, topk: list, metrics: list) -> Dict[str, float]:
    #     """
    #     :param predictions: (-1, n_candidates) shape, the first column is the score for ground-truth item
    #     :param topk: top-K value list
    #     :param metrics: metric string list
    #     :return: a result dict, the keys are metric@topk
    #     """
    #     evaluations = dict()
    #     # sort_idx = (-predictions).argsort(axis=1)
    #     # gt_rank = np.argwhere(sort_idx == 0)[:, 1] + 1
    #     # ↓ As we only have one positive sample, comparing with the first item will be more efficient.
    #     gt_rank = (predictions >= predictions[:, 0].reshape(-1, 1)).sum(axis=-1)
    #     # if (gt_rank!=1).mean()<=0.05: # maybe all predictions are the same
    #     # 	predictions_rnd = predictions.copy()
    #     # 	predictions_rnd[:,1:] += np.random.rand(predictions_rnd.shape[0], predictions_rnd.shape[1]-1)*1e-6
    #     # 	gt_rank = (predictions_rnd > predictions[:,0].reshape(-1,1)).sum(axis=-1)+1
    #     for k in topk:
    #         hit = (gt_rank <= k)
    #         for metric in metrics:
    #             key = '{}@{}'.format(metric, k)
    #             if metric == 'HR':
    #                 evaluations[key] = hit.mean()
    #             elif metric == 'NDCG':
    #                 evaluations[key] = (hit / np.log2(gt_rank + 1)).mean()
    #             else:
    #                 raise ValueError('Undefined evaluation metric: {}.'.format(metric))
    #     return evaluations

    def __init__(self, args, handler):
        # super().__init__(args)  # 确保调用父类的构造函数
        self.args = args  # 将 args 存储为实例变量
        self.handler = handler
        self.epoch = args.epoch
        self.check_epoch = args.check_epoch
        self.test_epoch = args.test_epoch
        self.early_stop = args.early_stop
        self.learning_rate = args.lr
        self.batch_size = args.batch_size
        self.eval_batch_size = args.eval_batch_size
        self.l2 = args.l2
        self.optimizer_name = args.optimizer
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory
        self.topk = [int(x) for x in args.topk.split(',')]
        self.metrics = [m.strip().upper() for m in args.metric.split(',')]
        self.main_metric = '{}@{}'.format(self.metrics[0], self.topk[0]) if not len(
            args.main_metric) else args.main_metric  # early stop based on main_metric
        self.main_topk = int(self.main_metric.split("@")[1])
        self.fixSteps = args.fixSteps
        self.ssl_reg = args.ssl_reg
        self.time = None  # will store [start_time, last_step_time]
        self.model = None
        self.opt = None
        self.masker = None
        self.sampler = None
        self.log_path = os.path.dirname(args.log_file)  # path to save predictions
        self.save_appendix = args.log_file.split("/")[-1].split(".")[0]  # appendix for prediction saving
        self.save_path = args.save_path

    # 计算时间差
    def _check_time(self, start=False):
        if self.time is None or start:
            self.time = [time()] * 2
            return self.time[0]
        tmp_time = self.time[1]
        self.time[1] = time()
        return self.time[1] - tmp_time

    # 判断是否早停
    def eval_termination(self, criterion: List[float]) -> bool:
        if len(criterion) > self.early_stop and utils.non_increasing(criterion[-self.early_stop:]):
            return True
        elif len(criterion) - criterion.index(max(criterion)) > self.early_stop:
            return True
        return False

    # # 评估输入数据集的结果，返回评估结果字典。
    # def evaluate(self, dataset: BaseModel.Dataset, topks: list, metrics: list) -> Dict[str, float]:
    #     """
    #     Evaluate the results for an input dataset.
    #     :return: result dict (key: metric@k)
    #     """
    #     predictions = self.predict(dataset)
    #     return self.evaluate_method(predictions, topks, metrics)

    def prepareModel(self, corpus):
        self.model = AutoCF_mat(args=self.args, corpus=corpus)  # 定义模型，传递 args 和 corpu
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=0)
        self.masker = RandomMaskSubgraphs(args=self.args)  # 定义 masker
        self.sampler = LocalGraph(args=self.args)  # 定义 sampler

    def trainEpoch(self, corpus):
        # print(f"Handler attributes: {dir(self.handler)}")  # 列出所有属性
        trnLoader = self.handler.trnLoader
        # print(f"corpus.n_users:{corpus.n_items}")  # 8714
        #########################################
        trnLoader.dataset.negSampling(corpus.n_items)
        # print(f"self.handler.torchBiAdj:{self.handler.torchBiAdj.shape}")
        epLoss, epPreLoss = 0, 0
        steps = trnLoader.dataset.__len__() // self.batch_size
        print("steps:", steps)
        for i, tem in enumerate(trnLoader):
            if i % self.fixSteps == 0:
                sampScores, seeds = self.sampler(self.handler.allOneAdj, self.model.getEgoEmbeds())
                # print(f"self.handler.torchBiAdj:{self.handler.torchBiAdj.shape}")
                encoderAdj, decoderAdj = self.masker(self.handler.torchBiAdj, seeds)
                # print(f"encoderAdj:{encoderAdj.shape}")


            ancs, poss, _ = tem  # 获取正样本
            ancs = ancs.long()
            poss = poss.long()
            usrEmbeds, itmEmbeds = self.model(encoderAdj, decoderAdj)  # 获取用户和物品嵌入
            # print("itmEmbeds shape:", itmEmbeds.shape)
            ancEmbeds = usrEmbeds[ancs]  # 用户嵌入
            posEmbeds = itmEmbeds[poss]  # 物品嵌入

            # 计算BPR损失
            bprLoss = (-torch.sum(ancEmbeds * posEmbeds, dim=-1)).mean()
            regLoss = calcRegLoss(self.model) * self.l2

            # 计算对比损失
            # contrastLoss = (contrast(ancs, usrEmbeds) + contrast(poss, itmEmbeds)) * self.ssl_reg
            ###################################
            contrastLoss = (contrast(ancs, usrEmbeds) + contrast(poss, itmEmbeds)) * self.ssl_reg + contrast(ancs,
                                                                                                             usrEmbeds,
                                                                                                             itmEmbeds)

            # 总损失
            loss = bprLoss + regLoss + contrastLoss

            if i % self.fixSteps == 0:
                localGlobalLoss = -sampScores.mean()
                loss += localGlobalLoss

            epLoss += loss.item()
            epPreLoss += bprLoss.item()

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            logging.info('Step %d/%d: loss = %.1f, reg = %.1f, cl = %.1f' % (i, steps, loss, regLoss, contrastLoss))

        return {"Loss": epLoss / steps, "preLoss": epPreLoss / steps}

    # def train(self, data_dict: Dict[str, BaseModel.Dataset]):
    #     print("args.user55555:", self.args.user)
    #     self.prepareModel(data_dict['train'].corpus)
    #     stloc = 0
    #     logging.info('Model Initialized')
    #     self._check_time(start=True)
    #     bestRes = None
    #     for ep in range(stloc, self.epoch):
    #         tstFlag = (ep % self.test_epoch == 0)
    #         reses = self.trainEpoch(data_dict['train'].corpus)
    #         logging.info(self.makePrint('Train', ep, reses, tstFlag))
    #         if tstFlag:
    #             reses = self.testEpoch()
    #             logging.info(self.makePrint('Test', ep, reses, tstFlag))
    #             self.saveHistory()
    #             bestRes = reses if bestRes is None or reses['Recall'] > bestRes['Recall'] else bestRes
    #         print()
    #     reses = self.testEpoch()
    #     logging.info(self.makePrint('Test', self.epoch, reses, True))
    #     logging.info(self.makePrint('Best Result', self.epoch, bestRes, True))
    #     self.saveHistory()

    def train(self, data_dict: Dict[str, BaseModel.Dataset]):
        self.prepareModel(data_dict['train'].corpus)
        stloc = 0
        logging.info('Model Initialized')
        self._check_time(start=True)
        bestRes = None

        for ep in range(stloc, self.epoch):
            tstFlag = (ep % self.test_epoch == 0)
            reses = self.trainEpoch(data_dict['train'].corpus)
            logging.info(self.makePrint('Train', ep, reses, tstFlag))

            if tstFlag:
                reses = self.testEpoch()
                logging.info(self.makePrint('Test', ep, reses, tstFlag))
                self.saveHistory()
                # 使用 HR@5 替代 Recall
                bestRes = reses if bestRes is None or reses['HR@5'] > bestRes['HR@5'] else bestRes

            print()

        # 最终测试
        reses = self.testEpoch()
        logging.info(self.makePrint('Test', self.epoch, reses, True))
        logging.info(self.makePrint('Best Result', self.epoch, bestRes, True))
        self.saveHistory()

    def predict(self, dataset: BaseModel.Dataset, save_prediction: bool = False) -> np.ndarray:
        dataset.model.eval()
        predictions = list()
        dl = DataLoader(dataset, batch_size=self.eval_batch_size, shuffle=False, num_workers=self.num_workers,
                        collate_fn=dataset.collate_batch, pin_memory=self.pin_memory)

        for batch in tqdm(dl, leave=False, ncols=100, mininterval=1, desc='Predict'):
            if hasattr(dataset.model, 'inference'):
                prediction = dataset.model.inference(utils.batch_to_gpu(batch, dataset.model.device))[0]
            else:
                usrEmbeds, itmEmbeds = dataset.model(utils.batch_to_gpu(batch, dataset.model.device))
                print("usrEmbeds:", usrEmbeds.shape)
                print("itmEmbeds:", itmEmbeds.shape)
                prediction = usrEmbeds  # 或 itmEmbeds，根据你的具体需求选择

            predictions.extend(prediction.cpu().data.numpy())

        predictions = np.array(predictions)

        if dataset.model.test_all:
            rows, cols = list(), list()
            for i, u in enumerate(dataset.data['user_id']):
                clicked_items = list(dataset.corpus.train_clicked_set[u] | dataset.corpus.residual_clicked_set[u])
                idx = list(np.ones_like(clicked_items) * i)
                rows.extend(idx)
                cols.extend(clicked_items)
            predictions[rows, cols] = -np.inf

        return predictions

    # def fit(self, dataset: BaseModel.Dataset, epoch=-1) -> float:
    #     model = dataset.model
    #     # if model.optimizer is None:
    #     #     model.optimizer = self._build_optimizer(model)
    #     dataset.actions_before_epoch()
    #
    #     model.train()
    #     loss_lst = list()
    #     dl = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
    #                     collate_fn=dataset.collate_batch, pin_memory=self.pin_memory)
    #
    #     for batch in tqdm(dl, leave=False, desc='Epoch {:<3}'.format(epoch), ncols=100, mininterval=1):
    #         batch = utils.batch_to_gpu(batch, model.device)
    #
    #         # 随机打乱 item_ids
    #         item_ids = batch['item_id']
    #         indices = torch.argsort(torch.rand(*item_ids.shape), dim=-1)
    #         batch['item_id'] = item_ids[torch.arange(item_ids.shape[0]).unsqueeze(-1), indices]
    #
    #         model.optimizer.zero_grad()
    #
    #         # 获取用户和物品嵌入
    #         usrEmbeds, itmEmbeds = model(batch)
    #
    #         # 计算所有预测值
    #         allPreds = torch.mm(usrEmbeds, torch.transpose(itmEmbeds, 1, 0))
    #
    #         # 计算损失
    #         out_dict = {'prediction': allPreds}
    #         loss = model.loss(out_dict)
    #         loss.backward()
    #         model.optimizer.step()
    #
    #         loss_lst.append(loss.detach().cpu().data.numpy())
    #
    #     return np.mean(loss_lst).item()

    def makePrint(self, name, ep, reses, save):
        ret = 'Epoch %d/%d, %s: ' % (ep, self.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += '%s = %.4f, ' % (metric, val)
            tem = name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        ret = ret[:-2] + '  '
        return ret

    def testEpoch(self):
        # tstLoader = self.handler.tstLoader
        # # print("tstLoader:", tstLoader)
        # epLoss, epRecall, epNdcg = [0] * 3
        # i = 0
        # num = tstLoader.dataset.__len__()
        # print("num:", num)
        # steps = num // self.args.tstBat
        # print("steps2:", steps)
        # for usr, trnMask in tstLoader:
        #     print(f"usr shape: {usr.shape}, trnMask shape: {trnMask.shape}")
        #     i += 1
        #     usr = usr.long()
        #     trnMask = trnMask
        #     usrEmbeds, itmEmbeds = self.model(self.handler.torchBiAdj, self.handler.torchBiAdj)
        #     print("usrEmbeds:", usrEmbeds[usr].shape)
        #     print("itmEmbeds:", itmEmbeds.shape)
        #     #########################################################
        #     allPreds = torch.mm(usrEmbeds[usr], torch.transpose(itmEmbeds, 1, 0)) * (1 - trnMask) - trnMask * 1e8
        #     # print("allPred:", allPreds.shape)
        #     _, topLocs = torch.topk(allPreds, int(self.args.topk[0]))
        #     recall, ndcg = self.calcRes(topLocs.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr)
        #     epRecall += recall
        #     epNdcg += ndcg
        #     logging.info('Steps %d/%d: recall = %.1f, ndcg = %.1f          ' % (i, steps, recall, ndcg))
        # ret = dict()
        # ret['Recall'] = epRecall / num
        # ret['NDCG'] = epNdcg / num
        # return ret
        tstLoader = self.handler.tstLoader
        epLoss, epRecall, epNdcg = [0] * 3
        i = 0
        num = tstLoader.dataset.__len__()
        print("num:", num)
        steps = num // self.args.tstBat
        print("steps2:", steps)

        for usr, trnMask in tstLoader:
            print(f"usr shape: {usr.shape}, trnMask shape: {trnMask.shape}")
            i += 1
            usr = usr.long()
            trnMask = trnMask

            usrEmbeds, itmEmbeds = self.model(self.handler.torchBiAdj, self.handler.torchBiAdj)
            print("usrEmbeds:", usrEmbeds[usr].shape)
            print("itmEmbeds:", itmEmbeds.shape)

            #########################################################
            allPreds = torch.mm(usrEmbeds[usr], torch.transpose(itmEmbeds, 1, 0)) * (1 - trnMask) - trnMask * 1e8
            _, topLocs = torch.topk(allPreds, int(self.args.topk[0]))
            print("args.topk:", int(self.args.topk[0]))
            hr_at_5, ndcg = self.calcRes(topLocs.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr)

            epRecall += hr_at_5
            epNdcg += ndcg
            logging.info('Steps %d/%d: HR@5 = %.1f, NDCG@5 = %.1f' % (i, steps, hr_at_5, ndcg))

        ret = dict()
        ret['HR@5'] = epRecall / num
        ret['NDCG@5'] = epNdcg / num
        return ret

    def calcRes(self, topLocs, tstLocs, batIds):
        # assert topLocs.shape[0] == len(batIds)
        # allRecall = allNdcg = 0
        # for i in range(len(batIds)):
        #     temTopLocs = list(topLocs[i])
        #     temTstLocs = tstLocs[batIds[i]]
        #     # print("temTstLocs:", temTstLocs, "Type:", type(temTstLocs))
        #     tstNum = len(temTstLocs)
        #     maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, int(self.args.topk[0])))])
        #     recall = dcg = 0
        #     for val in temTstLocs:
        #         if val in temTopLocs:
        #             recall += 1
        #             dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
        #     recall = recall / tstNum
        #     ndcg = dcg / maxDcg
        #     allRecall += recall
        #     allNdcg += ndcg
        # return allRecall, allNdcg

        assert topLocs.shape[0] == len(batIds)
        allHrAt5 = allNdcg = 0

        for i in range(len(batIds)):
            temTopLocs = list(topLocs[i][:5])  # 只考虑前5个推荐
            temTstLocs = tstLocs[batIds[i]]

            tstNum = len(temTstLocs)
            maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, 5))])
            dcg = 0

            # 计算 HR@5
            hr_at_5 = any(val in temTopLocs for val in temTstLocs)  # 检查是否有任一正确的物品
            hr_at_5 = 1.0 if hr_at_5 else 0.0  # 如果有命中则 HR@5 = 1

            # 计算 NDCG
            for val in temTstLocs:
                if val in temTopLocs:
                    dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))

            ndcg = dcg / maxDcg if maxDcg > 0 else 0.0

            allHrAt5 += hr_at_5
            allNdcg += ndcg

        return allHrAt5, allNdcg

    def saveHistory(self):
        if self.epoch == 0:
            return
        with open('History/' + self.save_path + '.his', 'wb') as fs:
            pickle.dump(self.metrics, fs)

        content = {
            'model': self.model,
        }
        torch.save(content, 'Models/' + self.save_path + '.mod')
        logging.info('Model Saved: %s' % self.save_path)