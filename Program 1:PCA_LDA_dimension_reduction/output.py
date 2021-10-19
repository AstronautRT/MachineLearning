#output.py为测试文件，仅用于测试数据集以及相关函数的输出
#还测试了.npy文件和.txt文件的相互转化

import numpy as np
input_data = np.load(r"feat_before_classifier_set.npy")
# input_data=np.load(r"feat_from_resnet_set.npy")
print(input_data.shape) #输出其形状
#reshape(m,n):将数据转化为m行n列；必须是矩阵格式或者数组格式，才能使用 .reshape(m, -1) 函数， 表示将此矩阵或者数组重组
# data = input_data.reshape(1,-1) #转化为1行
# print(data.shape) #转化为1行后的形状
# print(data)
print(input_data)

# 写入.txt文件
# file=open('D:\studyforgradethree\machinelearning\program1\label.txt','w') # 'w'是覆盖写 'a'是追加写 
# for i in range(data.shape[0]):
#     for j in range(data.shape[1]):
#         file.write(str(data[i][j])+" ")
# file.close();
