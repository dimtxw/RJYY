# 数据预处理与采样是在线的，没有专门的步骤

# 模型训练
python train.py -k 5 -v 1

# round2测试集B预测，并生成结果文件
python inference.py

# 结果文件生成在当前目录的submit目录下，没有打包