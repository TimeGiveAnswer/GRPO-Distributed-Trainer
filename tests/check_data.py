import pandas as pd

# 读取刚刚生成的训练集
df = pd.read_parquet("data/datasets/MATH-lighteval2/train.parquet")

# 打印前 2 条看看字段
print("Total examples:", len(df))
print("\nColumns:", df.columns.tolist())
print("-" * 30)

# 打印第一条数据的内容
first_row = df.iloc[0]
print("Prompt Content:", first_row['prompt'][0]['content'])
print("Ground Truth:", first_row['reward_model']['ground_truth'])