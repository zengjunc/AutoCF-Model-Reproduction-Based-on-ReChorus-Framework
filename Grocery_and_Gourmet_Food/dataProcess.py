import logging
import numpy as np
import pickle
from scipy.sparse import csr_matrix
import time

# 时间转换函数，转换时间戳为时间并记录最小和最大年份
minn = 2022
maxx = 0

def transTime(timeStamp):
    timeArr = time.localtime(timeStamp)
    year = timeArr.tm_year
    global minn
    global maxx
    minn = min(minn, year)
    maxx = max(maxx, year)
    return time.mktime(timeArr)

# 数据映射函数，用于读取数据并初步生成交互矩阵
def mapping(infile):
    usrId = dict()
    itmId = dict()
    usrid, itmid = [0, 0]
    interaction = list()
    with open(infile, 'r') as fs:
        next(fs)  # 跳过标题行
        for line in fs:
            arr = line.strip().split('\t')
            row = arr[0]
            col = arr[1]
            timeStamp = transTime(int(arr[-1]))
            if timeStamp is None:
                continue
            if row not in usrId:
                usrId[row] = usrid
                interaction.append(dict())
                usrid += 1
            if col not in itmId:
                itmId[col] = itmid
                itmid += 1
            usr = usrId[row]
            itm = itmId[col]
            interaction[usr][itm] = timeStamp
    return interaction, usrid, itmid, usrId, itmId

# 重新编号函数，确保用户和物品ID连续
def remap_ids(interaction, usrnum, itmnum):
    usrIdMap = {old_id: new_id for new_id, old_id in enumerate(range(usrnum))}
    itmIdMap = {old_id: new_id for new_id, old_id in enumerate(range(itmnum))}
    remapped_interaction = [{} for _ in range(usrnum)]
    for old_usr, data in enumerate(interaction):
        new_usr = usrIdMap.get(old_usr, None)
        if new_usr is not None:
            for old_itm, value in data.items():
                new_itm = itmIdMap.get(old_itm, None)
                if new_itm is not None:
                    remapped_interaction[new_usr][new_itm] = value
    return remapped_interaction, usrIdMap, itmIdMap

# 划分数据集，将部分数据划分为测试集
def split(interaction, usrnum, itmnum):
    pickNum = 10000
    usrPerm = np.random.permutation(usrnum)
    pickUsr = usrPerm[:pickNum]

    tstInt = [None] * usrnum
    exception = 0
    for usr in pickUsr:
        temp = list()
        data = interaction[usr]
        for itm in data:
            temp.append((itm, data[itm]))
        if len(temp) == 0:
            exception += 1
            continue
        temp.sort(key=lambda x: x[1])
        tstInt[usr] = temp[-1][0]
        interaction[usr][tstInt[usr]] = None
    print('Exception:', exception, np.sum(np.array(tstInt) != None))
    return interaction, tstInt

# 将交互数据转换为稀疏矩阵
def trans(interaction, usrnum, itmnum):
    r, c, d = [list(), list(), list()]
    for usr in range(usrnum):
        if interaction[usr] is None:
            continue
        data = interaction[usr]
        for col in data:
            if data[col] is not None:
                r.append(usr)
                c.append(col)
                d.append(data[col])
    intMat = csr_matrix((d, (r, c)), shape=(usrnum, itmnum))
    return intMat

# 主程序入口
prefix = './'
logging.info('Start')
interaction, usrnum, itmnum, usrId, itmId = mapping(prefix + 'combined_data.csv')

# 在映射后进行重新编号
interaction, usrIdMap, itmIdMap = remap_ids(interaction, usrnum, itmnum)
print('IDs remapped, usr %d, itm %d' % (usrnum, itmnum))

# 划分训练和测试集
trnInt, tstInt = split(interaction, usrnum, itmnum)
logging.info('Datasets Split')

# 转换为稀疏矩阵
trnMat = trans(trnInt, usrnum, itmnum)
print(len(trnMat.data))
logging.info('Train Matrix Done')

# 保存训练和测试数据
with open(prefix + 'trn_mat', 'wb') as fs:
    pickle.dump(trnMat, fs)
with open(prefix + 'tst_int', 'wb') as fs:
    pickle.dump(tstInt, fs)
logging.info('Interaction Data Saved')
