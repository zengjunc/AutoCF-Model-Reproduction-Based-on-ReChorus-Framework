import pickle
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
import scipy.sparse as sp
import torch as t
import torch.utils.data as data
import torch.utils.data as dataloader
import logging
import os
import pandas as pd
from utils import utils


class Base(object):
    @staticmethod
    def parse_data_args(parser):
        parser.add_argument('--path', type=str, default='data/', help='Input data dir.')
        parser.add_argument('--dataset', type=str, default='Grocery_and_Gourmet_Food', help='Choose a dataset.')
        parser.add_argument('--sep', type=str, default='\t', help='Separator of CSV file.')
        return parser

    def __init__(self, args):
        self.sep = args.sep
        self.prefix = args.path
        self.dataset = args.dataset
        self._read_data()

        self.train_clicked_set = dict()  # store clicked item set in training
        self.residual_clicked_set = dict()  # store residual clicked items for dev/test

        for key in ['train', 'dev', 'test']:
            df = self.data_df[key]
            for uid, iid in zip(df['user_id'], df['item_id']):
                if uid not in self.train_clicked_set:
                    self.train_clicked_set[uid] = set()
                    self.residual_clicked_set[uid] = set()
                if key == 'train':
                    self.train_clicked_set[uid].add(iid)
                else:
                    self.residual_clicked_set[uid].add(iid)

    def _read_data(self):
        logging.info('Reading data from \"{}\", dataset = \"{}\" '.format(self.prefix, self.dataset))
        self.data_df = dict()
        for key in ['train', 'dev', 'test']:
            file_path = os.path.join(self.prefix, self.dataset, key + '.csv').replace('\\', '/')
            self.data_df[key] = pd.read_csv(file_path, sep=self.sep).reset_index(drop=True).sort_values(
                by=['user_id', 'time'])
            self.data_df[key] = utils.eval_list_columns(self.data_df[key])

        logging.info('Counting dataset statistics...')
        key_columns = ['user_id', 'item_id', 'time']
        if 'label' in self.data_df['train'].columns:
            key_columns.append('label')
        self.all_df = pd.concat([self.data_df[key][key_columns] for key in ['train', 'dev', 'test']])
        self.n_users, self.n_items = self.all_df['user_id'].max() + 1, self.all_df['item_id'].max() + 1

        for key in ['dev', 'test']:
            if 'neg_items' in self.data_df[key]:
                neg_items = np.array(self.data_df[key]['neg_items'].tolist())
                assert (neg_items >= self.n_items).sum() == 0
        logging.info('"# user": {}, "# item": {}, "# entry": {}'.format(
            self.n_users - 1, self.n_items - 1, len(self.all_df)))
        if 'label' in key_columns:
            positive_num = (self.all_df.label == 1).sum()
            logging.info('"# positive interaction": {} ({:.1f}%)'.format(
                positive_num, positive_num / self.all_df.shape[0] * 100))


class AutoCFReader_mat(Base):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        # Initialize file paths based on dataset
        if args.dataset == 'Grocery_and_Gourmet_Food':
            predir = 'data/Grocery_and_Gourmet_Food/'
        elif args.dataset == 'MIND_Large/MINDTOPK':
            predir = 'data/MIND_Large/MINDTOPK/'
        elif args.dataset == 'MovieLens-1M/ML_1MTOPK':
            predir = 'data/MovieLens-1M/ML_1MTOPK/'
        self.predir = predir
        self.trnfile = predir + 'trnMat.pkl'
        self.tstfile = predir + 'tstMat.pkl'
        self.trnMat = self.loadTrainFile()
        self.tstLst = self.loadTestFile()
        # Get user and item counts
        self.n_users, self.n_items = self.trnMat.shape
        args.user, args.item = self.trnMat.shape

    def loadTrainFile(self):
        with open(self.trnfile, 'rb') as fs:
            ret = (pickle.load(fs) != 0).astype(np.float32)
        if not isinstance(ret, coo_matrix):
            ret = sp.coo_matrix(ret)
        return ret

    def loadTestFile(self):
        with open(self.tstfile, 'rb') as fs:
            ret = (pickle.load(fs) != 0).astype(np.float32)
        if not isinstance(ret, coo_matrix):
            ret = sp.coo_matrix(ret)
        return ret.tocsr()

    def normalizeAdj(self, mat):
        degree = np.array(mat.sum(axis=-1))
        dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
        dInvSqrt[np.isinf(dInvSqrt)] = 0.0
        dInvSqrtMat = sp.diags(dInvSqrt)
        return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

    def makeTorchAdj(self, mat):
        a = sp.csr_matrix((self.args.user, self.args.user))
        b = sp.csr_matrix((self.args.item, self.args.item))
        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat != 0) * 1.0
        mat = (mat + sp.eye(mat.shape[0])) * 1.0
        mat = self.normalizeAdj(mat)

        idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = t.from_numpy(mat.data.astype(np.float32))
        shape = t.Size(mat.shape)
        return t.sparse.FloatTensor(idxs, vals, shape)

    def makeAllOne(self, torchAdj):
        idxs = torchAdj._indices()
        vals = t.ones_like(torchAdj._values())
        shape = torchAdj.shape
        return t.sparse.FloatTensor(idxs, vals, shape)

    def LoadData(self, args):
        trnMat = self.loadTrainFile()
        tstLst = self.loadTestFile()
        args.user, args.item = trnMat.shape
        self.torchBiAdj = self.makeTorchAdj(trnMat)
        self.allOneAdj = self.makeAllOne(self.torchBiAdj)

        trnData = TrnData(trnMat)
        self.trnLoader = dataloader.DataLoader(trnData, batch_size=4096, shuffle=True, num_workers=0)
        tstData = TstData(tstLst, trnMat)
        self.tstLoader = dataloader.DataLoader(tstData, batch_size=256, shuffle=False, num_workers=0)

        # Correctly handle data for training, validation, and testing
        # trnMat = coo_matrix(trnMat)
        # row, col, data = trnMat.row, trnMat.col, trnMat.data
        # row, col, data = np.array(row), np.array(col), np.array(data)
        # indices = np.random.permutation(len(row))
        # trn_end = int(len(row) * 0.7)
        # val_end = int(len(row) * 0.75)
        #
        # trn_indices = indices[:trn_end]
        # val_indices = indices[trn_end:val_end]
        # tst_indices = indices[val_end:]
        #
        # trnMat = coo_matrix((data[trn_indices], (row[trn_indices], col[trn_indices])), shape=[args.user, args.item])
        # valMat = coo_matrix((data[val_indices], (row[val_indices], col[val_indices])), shape=[args.user, args.item])
        # tstMat = coo_matrix((data[tst_indices], (row[tst_indices], col[tst_indices])), shape=[args.user, args.item])
        #
        # with open(f'{self.predir}trnMat.pkl', 'wb') as fs:
        #     pickle.dump(trnMat, fs)
        # with open(f'{self.predir}valMat.pkl', 'wb') as fs:
        #     pickle.dump(valMat, fs)
        # with open(f'{self.predir}tstMat.pkl', 'wb') as fs:
        #     pickle.dump(tstMat, fs)



class TrnData(data.Dataset):
    def __init__(self, coomat):
        self.rows = coomat.row
        self.cols = coomat.col
        self.dokmat = coomat.todok()
        self.negs = np.zeros(len(self.rows)).astype(np.int32)

    def negSampling(self, item_num):
        for i in range(len(self.rows)):
            u = self.rows[i]
            while True:
                iNeg = np.random.randint(item_num)
                if (u, iNeg) not in self.dokmat:
                    break
            self.negs[i] = iNeg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx], self.cols[idx], self.negs[idx]


class TstData(data.Dataset):
    def __init__(self, coomat, trnMat):
        # 将 coomat 转换为 COO 格式
        coomat = coomat.tocoo()
        self.csrmat = (trnMat.tocsr() != 0) * 1.0

        tstLocs = [None] * coomat.shape[0]
        tstUsrs = set()
        for i in range(len(coomat.data)):
            row = coomat.row[i]
            col = coomat.col[i]
            if tstLocs[row] is None:
                tstLocs[row] = list()
            tstLocs[row].append(col)
            tstUsrs.add(row)
        tstUsrs = np.array(list(tstUsrs))
        self.tstUsrs = tstUsrs
        self.tstLocs = tstLocs

    def __len__(self):
        return len(self.tstUsrs)

    def __getitem__(self, idx):
        return self.tstUsrs[idx], np.reshape(self.csrmat[self.tstUsrs[idx]].toarray(), [-1])
