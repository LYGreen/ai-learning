import pandas as pd

# 将文件路径保存到变量中以便轻松访问
melbourne_file_path = 'input/melb_data.csv'
# 读取数据并将其存储在名为 melbourne_data 的 DataFrame 中
melbourne_data = pd.read_csv(melbourne_file_path) 
# 打印墨尔本数据的摘要
print(melbourne_data.describe())