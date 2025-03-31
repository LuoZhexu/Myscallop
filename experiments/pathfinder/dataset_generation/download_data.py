import os
import kagglehub

# 临时指定 kaggle.json 的位置
os.environ["KAGGLE_CONFIG_DIR"] = "/home/luozx/scallop/experiments/pathfinder/dataset_generation"

# 下载数据
path = kagglehub.dataset_download("hajarbel04/pathfinder32")
print("Downloaded to:", path)
