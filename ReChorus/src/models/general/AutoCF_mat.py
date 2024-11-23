import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import utils
from models.BaseModel import GeneralModel
import argparse

# 初始化定义
init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class AutoCF_mat(GeneralModel):
    reader = 'AutoCFReader_mat'
    runner = 'AutoCFRunner_mat'
    extra_log_args = ['latdim', 'gcn_layers', 'gt_layers', 'head', 'seedNum', 'maskDepth', 'keepRate']

    @staticmethod
    def parse_model_args(parser):
        """只为 AutoCF 添加特定参数"""
        parser.add_argument('--latdim', default=32, type=int, help='embedding size')
        parser.add_argument('--head', default=4, type=int, help='number of heads in attention')
        parser.add_argument('--gcn_layers', default=2, type=int, help='number of gcn layers')
        parser.add_argument('--gt_layers', default=1, type=int, help='number of graph transformer layers')
        parser.add_argument('--seedNum', default=500, type=int, help='number of seeds in patch masking')
        parser.add_argument('--maskDepth', default=2, type=int, help='depth to mask')
        parser.add_argument('--keepRate', default=0.2, type=float, help='ratio of nodes to keep')
        # 调用父类的通用参数解析
        return GeneralModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.args = args  # 保存 args
        print("args.user0000:", args.user)
        print(f"self.user_num:{self.user_num}")
        self.uEmbeds = nn.Parameter(init(torch.empty(self.user_num, args.latdim)))
        self.iEmbeds = nn.Parameter(init(torch.empty(self.item_num, args.latdim)))
        self.gcnLayers = nn.Sequential(*[GCNLayer(args) for i in range(args.gcn_layers)])  # 传递 args
        self.gtLayers = nn.Sequential(*[GTLayer(args) for i in range(args.gt_layers)])  # 传递 args
        self._define_params()
        self.apply(self.init_weights)

    def getEgoEmbeds(self):
        print("用户嵌入形状:", self.uEmbeds.shape)
        print("物品嵌入形状:", self.iEmbeds.shape)

        return torch.concat([self.uEmbeds, self.iEmbeds], axis=0)

    def forward(self, encoderAdj, decoderAdj=None):
        # print(f"encoderAdj: {encoderAdj}")
        # encoderAdj type: <class 'dict'>
        if isinstance(encoderAdj, dict):
            # print(f"encoderAdj: {encoderAdj}")
            user_ids = encoderAdj['user_id'].long()  # 转换为 LongTensor
            item_ids = encoderAdj['item_id'].long()  # 转换为 LongTensor
            print(f"User IDs range: [{user_ids.min()}, {user_ids.max()}]")
            print(f"Item IDs range: [{item_ids.min()}, {item_ids.max()}]")
            # 展平 item_ids
            item_ids = item_ids.flatten()  # 形状为 [25600]

            # 将 user_ids 重复，确保与 item_ids 的大小匹配
            user_ids = user_ids.repeat_interleave(item_ids.shape[0] // user_ids.shape[0])

            # 确保 user_ids 和 item_ids 的形状一致
            assert user_ids.shape[0] == item_ids.shape[
                0], f"user_ids shape {user_ids.shape}, item_ids shape {item_ids.shape}"

            # print("args.user44444:", self.args.user)
            indices = torch.stack([user_ids, item_ids])  # 形状为 (2, N)
            print("indices:", indices.shape)
            values = torch.ones(indices.shape[1])  # 创建 values 张量，假设所有值为 1
            print("values:", values.shape)
            # 构建稀疏矩阵
            encoderAdj = torch.sparse.FloatTensor(indices, values,
                                                  (self.args.user + self.args.item, self.args.user + self.args.item))

        # print(f"encoderAdj type: {type(encoderAdj)}")

        # 处理 decoderAdj
        if decoderAdj is not None and isinstance(decoderAdj, dict):
            indices = torch.tensor(list(decoderAdj.keys())).t()
            values = torch.tensor(list(decoderAdj.values()))
            decoderAdj = torch.sparse.FloatTensor(indices, values,
                                                  (self.args.user + self.args.item, self.args.user + self.args.item))
        args = self.args  # 直接使用 self.args
        embeds = torch.concat([self.uEmbeds, self.iEmbeds], axis=0)
        embedsLst = [embeds]
        for i, gcn in enumerate(self.gcnLayers):
            # print(f"i:{i}")
            # print(f"gcn{gcn}")
            embeds = gcn(encoderAdj, embedsLst[-1])
            embedsLst.append(embeds)
        if decoderAdj is not None:
            for gt in self.gtLayers:
                embeds = gt(decoderAdj, embedsLst[-1])
                embedsLst.append(embeds)
        embeds = sum(embedsLst)
        ############################################################
        return embeds[:self.user_num], embeds[self.user_num:]
        # embeds = torch.concat([self.uEmbeds, self.iEmbeds], axis=0)
        # embedsLst = [embeds]
        # for i, gcn in enumerate(self.gcnLayers):
        #     embeds = gcn(encoderAdj, embedsLst[-1])
        #     embedsLst.append(embeds)
        # if decoderAdj is not None:
        #     for gt in self.gtLayers:
        #         embeds = gt(decoderAdj, embedsLst[-1])
        #         embedsLst.append(embeds)
        # embeds = sum(embedsLst)
        # return embeds[:self.user_num], embeds[self.user_num:]

class GCNLayer(nn.Module):
    def __init__(self, args):
        super(GCNLayer, self).__init__()

    def forward(self, adj, embeds):
        # print(f"adj shape: {adj.shape}")
        # print(f"embeds shape: {embeds.shape}")
        return torch.spmm(adj, embeds)

class GTLayer(nn.Module):
    def __init__(self, args):
        super(GTLayer, self).__init__()
        self.args = args  # 保存 args
        # print("args.user1111111:", args.user)
        self.qTrans = nn.Parameter(init(torch.empty(args.latdim, args.latdim)))
        self.kTrans = nn.Parameter(init(torch.empty(args.latdim, args.latdim)))
        self.vTrans = nn.Parameter(init(torch.empty(args.latdim, args.latdim)))

    # adj: 图的邻接矩阵，通常是稀疏的, embeds: 这是节点的嵌入（表示）
    def forward(self, adj, embeds):
        args = self.args  # 直接使用 self.args
        indices = adj._indices()
        rows, cols = indices[0, :], indices[1, :]  # 获取邻接矩阵中的边索引，rows 和 cols 分别表示边的起点和终点的节点索引。
        rowEmbeds = embeds[rows]
        colEmbeds = embeds[cols]

        # 线性变换得到多头注意力形式的q、k、v嵌入
        qEmbeds = (rowEmbeds @ self.qTrans).view([-1, args.head, args.latdim // args.head])
        kEmbeds = (colEmbeds @ self.kTrans).view([-1, args.head, args.latdim // args.head])
        vEmbeds = (colEmbeds @ self.vTrans).view([-1, args.head, args.latdim // args.head])

        # 注意力权重计算
        att = torch.einsum('ehd, ehd -> eh', qEmbeds, kEmbeds)  # 注意力分数
        att = torch.clamp(att, -10.0, 10.0)  # 对注意力分数进行裁剪，防止数值过大或过小。
        expAtt = torch.exp(att)  # 取对数
        tem = torch.zeros([adj.shape[0], args.head])
        attNorm = (tem.index_add_(0, rows, expAtt))[rows]
        att = expAtt / (attNorm + 1e-8)  # eh 归一化注意力权重，防止分母为零。

        # 基于注意力的节点嵌入更新
        resEmbeds = torch.einsum('eh, ehd -> ehd', att, vEmbeds).view([-1, args.latdim])
        tem = torch.zeros([adj.shape[0], args.latdim])
        resEmbeds = tem.index_add_(0, rows, resEmbeds)  # nd

        # 最终返回的是经过图注意力机制更新后的节点嵌入。
        return resEmbeds

class LocalGraph(nn.Module):
    def __init__(self, args):
        super(LocalGraph, self).__init__()
        self.args = args  # 保存 args
        # print("args.user2222222:", args.user)

    def makeNoise(self, scores):
        noise = torch.rand(scores.shape)
        noise = -torch.log(-torch.log(noise))
        return torch.log(scores) + noise

    def forward(self, allOneAdj, embeds):
        args = self.args  # 使用 self.args
        # 计算邻接矩阵每个节点的度（连接的边数），将稀疏张量转换为稠密格式并调整形状。
        order = torch.sparse.sum(allOneAdj, dim=-1).to_dense().view([-1, 1])
        # 计算首次和二次嵌入,无误
        # print(f"allOneAdj shape: {allOneAdj.shape}")
        # print(f"embeds shape: {embeds.shape}")
        fstEmbeds = torch.spmm(allOneAdj, embeds) - embeds
        fstNum = order
        scdEmbeds = (torch.spmm(allOneAdj, fstEmbeds) - fstEmbeds) - order * embeds
        scdNum = (torch.spmm(allOneAdj, fstNum) - fstNum) - order
        # 计算子图嵌入
        subgraphEmbeds = (fstEmbeds + scdEmbeds) / (fstNum + scdNum + 1e-8)
        subgraphEmbeds = F.normalize(subgraphEmbeds, p=2)
        embeds = F.normalize(embeds, p=2)
        # 计算得分
        scores = torch.sigmoid(torch.sum(subgraphEmbeds * embeds, dim=-1))
        # 对得分添加噪声
        scores = self.makeNoise(scores)
        # 选择得分最高的 seedNum 个节点作为种子节点。
        _, seeds = torch.topk(scores, args.seedNum)
        return scores, seeds


class RandomMaskSubgraphs(nn.Module):
    def __init__(self, args):
        super(RandomMaskSubgraphs, self).__init__()
        self.args = args  # 保存 args
        # print("args.user33333:", args.user)
        self.flag = False  # 是否打印信息

    # 邻接矩阵归一化方法
    def normalizeAdj(self, adj):
        degree = torch.pow(torch.sparse.sum(adj, dim=1).to_dense() + 1e-12, -0.5)
        newRows, newCols = adj._indices()[0, :], adj._indices()[1, :]
        rowNorm, colNorm = degree[newRows], degree[newCols]
        newVals = adj._values() * rowNorm * colNorm
        return torch.sparse.FloatTensor(adj._indices(), newVals, adj.shape)

    def forward(self, adj, seeds):
        args = self.args  # 使用 self.args

        # 子图抽样
        rows = adj._indices()[0, :]
        cols = adj._indices()[1, :]

        maskNodes = [seeds]

        for i in range(args.maskDepth):
            curSeeds = seeds if i == 0 else nxtSeeds
            nxtSeeds = list()
            for seed in curSeeds:
                rowIdct = (rows == seed)
                colIdct = (cols == seed)
                idct = torch.logical_or(rowIdct, colIdct)

                if i != args.maskDepth - 1:
                    mskRows = rows[idct]
                    mskCols = cols[idct]
                    nxtSeeds.append(mskRows)
                    nxtSeeds.append(mskCols)

                rows = rows[torch.logical_not(idct)]
                cols = cols[torch.logical_not(idct)]
            if len(nxtSeeds) > 0:
                nxtSeeds = torch.unique(torch.concat(nxtSeeds))
                maskNodes.append(nxtSeeds)

        # 抽样节点
        print("args.user_randomMask:", args.user)
        print("args.item_randomMask:", args.item)
        sampNum = int((args.user + args.item) * args.keepRate)
        sampedNodes = torch.randint(args.user + args.item, size=[sampNum])

        if self.flag == False:
            l1 = adj._values().shape[0]
            l2 = rows.shape[0]
            print('-----')
            print('LENGTH CHANGE', '%.2f' % (l2 / l1), l2, l1)
            tem = torch.unique(torch.concat(maskNodes))
            print('Original SAMPLED NODES', '%.2f' % (tem.shape[0] / (args.user + args.item)), tem.shape[0],
                  (args.user + args.item))

        # 生成新的邻接矩阵
        maskNodes.append(sampedNodes)
        maskNodes = torch.unique(torch.concat(maskNodes))
        if self.flag == False:
            print('AUGMENTED SAMPLED NODES', '%.2f' % (maskNodes.shape[0] / (args.user + args.item)),
                  maskNodes.shape[0], (args.user + args.item))
            self.flag = True
            print('-----')

        # 构造解码器邻接矩阵
        encoderAdj = self.normalizeAdj(
            torch.sparse.FloatTensor(torch.stack([rows, cols], dim=0), torch.ones_like(rows), adj.shape))

        temNum = maskNodes.shape[0]
        temRows = maskNodes[torch.randint(temNum, size=[adj._values().shape[0]])]
        temCols = maskNodes[torch.randint(temNum, size=[adj._values().shape[0]])]

        newRows = torch.concat([temRows, temCols, torch.arange(args.user + args.item), rows])
        newCols = torch.concat([temCols, temRows, torch.arange(args.user + args.item), cols])

        # filter duplicated
        hashVal = newRows * (args.user + args.item) + newCols
        hashVal = torch.unique(hashVal)
        newCols = hashVal % (args.user + args.item)
        newRows = ((hashVal - newCols) / (args.user + args.item)).long()

        decoderAdj = torch.sparse.FloatTensor(torch.stack([newRows, newCols], dim=0), torch.ones_like(newRows).float(),
                                              adj.shape)
        # 返回编码器邻接矩阵 encoderAdj 和解码器邻接矩阵 decoderAdj

        return encoderAdj, decoderAdj
