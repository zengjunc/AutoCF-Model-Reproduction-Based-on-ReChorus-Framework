import pandas as pd

# 读取数据
train_df = pd.read_csv('train.csv', sep='\t')
test_df = pd.read_csv('test.csv', sep='\t')
dev_df = pd.read_csv('dev.csv', sep='\t')

# 去掉 test 和 dev 数据集中的 'neg_items' 列
test_df = test_df.drop(columns=['neg_items'])
dev_df = dev_df.drop(columns=['neg_items'])

# 选择需要的列
train_df = train_df[['user_id', 'item_id', 'time']]
test_df = test_df[['user_id', 'item_id', 'time']]
dev_df = dev_df[['user_id', 'item_id', 'time']]

# 合并数据
combined_df = pd.concat([train_df, test_df, dev_df], ignore_index=True)

# 保存为 CSV 文件，使用空格作为分隔符
combined_df.to_csv('combined_data.csv', sep='\t', index=False)
