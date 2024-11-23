# AutoCF Model Reproduction Based on ReChorus Framework

本项目旨在复现基于 ReChorus 框架的 AutoCF 模型。ReChorus 是一个开源的推荐系统框架，用于处理多种推荐算法的研究和复现工作。AutoCF 是一个自动化协同过滤模型，具有高效的用户和物品交互学习能力。

For more information, visit my [GitHub Page](https://github.com/zengjunc).

## Requirements

由于本人是在window11系统上进行复现，所用依赖库有所不同，以下列出与`requirements.txt`不同之处，以供参考
- Python 3.10.4
- NumPy 1.23.5
- SciPy 1.14.1
- pandas 1.5.3
- cudatoolkit 10.2.89 (此版本在pip上没有，需要使用conda)
- 其他依赖库请参考 `requirements.txt`

## Project Structure

- `src/`: 包含模型实现代码
  - `mainCF.py`: 复现任务的主程序
  - `AutoCF_mat.py`: AutoCF 模型定义
  - `AutoCFReader_mat.py`: 数据读取与预处理
  - `AutoCFRunner_mat.py`: 模型训练与评估
- `data/`: 数据文件存放目录
  - 1`Gowalla/`: Gowalla数据集
  - 2`Grocery_and_Gourmet_Food/`: amazon数据集
  - 3`MIND_Large/MIND_small`: MIND_small数据集(large数据太庞大了，受限于设备)
  - 4`MovieLens-1M`: MovieLens-1M数据集
  - 5`MovieLens-20M`: MovieLens-20M数据集
- `results/`: 训练结果和日志存放目录
- `README_mywork.md`: 项目说明文档

其中，由于模型部分不兼容，复现模型的结果保存在Models和History

效果对比只使用了2-4数据集，但其它数据集也进行了数据预处理，可以作为额外的数据测试

## Data Preparation

1. 下载并解压数据集（例如 ml-20m, Gowalla）。
2. 将数据放入 `data/` 目录下。
3. 运行相应的xxxx.iypnb进行数据预处理(数据集彼此间有差距)
4. 运行conbine_data获得数据总集，保证效果对比时各模型训练数据量一致
5. 对于autoCF模型，还需进行dataprocess处理得到trn_mat,tst_int，再通过mat2pkl转化成trnMat.pkl、tstMat.pkl、valMat.pkl(此步骤的具体做法请见`Usage`)

## Usage

运行下面的命令：
```
python src/mainCF2.py --model_name AutoCF_mat --emb_size 32 --lr 1e-3 --l2 1e-6 --dataset MovieLens-1M/ML_1MTOPK
```

- `--emb_size`: 批处理大小
- `--dataset`: 数据集名称
- `--lr`: 学习率
- `--l2`: 优化器的权重衰减
- `--seed`: 随机种子（推荐设置为500）
- 根据具体需要，也可以添加其它参数

## Result

| Data                     | Metric                | AutoCF                 | LightGCN               | FPMC                   | SLRPlus                | GRU4Rec                | NeuMF                  |
|:-------------------------|:----------------------|------------------------|------------------------|------------------------|------------------------|------------------------|------------------------|
| Grocery_and_Gourmet_Food | HR@5</br><br/>NDCG@5  | 0.1121</br><br/>0.0465 | 0.3858</br><br/>0.2659 | 0.3409</br><br/>0.2606 | 0.3242</br><br/>0.2249 | 0.3682</br><br/>0.2616 | 0.3261</br><br/>0.2242 |
| MIND_Large               | HR@5</br><br/>NDCG@5  | 0.2537</br><br/>0.0807 | 0.1078</br><br/>0.0631 | 0.1804</br><br/>0.1207 | 0.1098</br><br/>0.0716 | 0.2010</br><br/>0.1221 | 0.1020</br><br/>0.0638 |
| MovieLens-1M             | HR@5</br><br/>NDCG@5  | 0.6763</br><br/>0.2832 | 0.3520</br><br/>0.2382 | 0.4181</br><br/>0.2939 | 0.3693</br><br/>0.2455 | 0.4167</br><br/>0.2859 | 0.3319</br><br/>0.2277 |

分析：

AutoCF在 MIND Large、MovieLens-1M 数据集上表现基本优于对比的大部分模型，取得较好的效果，但是在 Grocery_and Gourmet Food 上效果非常差，经研究发现是数据稀疏性问题，通过再次过滤（缩小了数据集），能够达到HR@5:0.3140,NDCG@5:0.1124的正常水平。

## License

This project is licensed under the MIT License. It references ideas and methodologies from the following projects:

- **[Rechorus](https://github.com/THUwangcy/ReChorus)**
- **[AutoCF](https://github.com/HKUDS/AutoCF)**

Please refer to the respective project repositories for their specific licenses.
