# -*- coding: utf-8 -*-#
# Author:  KKKK0927
# Date:    2021-10-12
# project: MachineLearning/PCA_LDA_DIMENSION_REDUCTION
# Name:    pca_lda_dimension_reduction.py

##用于3D可视化
from mpl_toolkits.mplot3d import Axes3D
##用于可视化图表
import matplotlib.pyplot as plt
##用于做科学计算
import numpy as np
##导入PCA库
from sklearn.decomposition import PCA
##导入LDA库
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# font = {'family':'MicroSoft YaHei','weight':'bold'}
# matplotlib.rc('font',**font)  #使matplotlib显示中文

#加载数据集
label=np.load("label.npy")                                  #标签信息
resnet_set=np.load("feat_from_resnet_set.npy")              #用于LDA降维
classifier_set=np.load("feat_before_classifier_set.npy")    #用于PCA降维

#形状重构
#根据测试文件output.py的输出结果，可以了解到：feat_before_classifier_set.npy以及
#   feat_from_resnet_set.npy数据集的输出结果以三维数组的方式呈现
#但是：LinearDiscriminantAnalysis expected <= 2.
resnet_reshape=resnet_set.reshape(720,8*1024)
classifier_reshape=classifier_set.reshape(720,1*1024)

#加载PCA模型并训练，降维
#注意：PCA为无监督学习 无法使用类别信息来降维
model_pca=PCA(n_components=2)
X_pca=model_pca.fit_transform(classifier_reshape)                   #降维后的数据
X_pca=model_pca.fit_transform(resnet_reshape)

model_pca=PCA(n_components=3)
X_pca=model_pca.fit_transform(resnet_reshape,label)
X_pca=model_pca.fit_transform(classifier_reshape,label)

#加载LDA模型并训练，降维
#LDA为监督学习 需要使用标签信息
model_lda=LinearDiscriminantAnalysis(n_components=2)
X_lda=model_lda.fit_transform(resnet_reshape,label)
X_lda=model_lda.fit_transform(classifier_reshape,label)

model_lda=LinearDiscriminantAnalysis(n_components=3)
X_lda=model_lda.fit_transform(resnet_reshape,label)
X_lda=model_lda.fit_transform(classifier_reshape,label)

print("各主成分的方差值：",model_lda.explained_variance_)            #打印方差
print("各主成分的方差贡献率：",model_lda.explained_variance_ratio_)  #打印方差贡献率

#绘图
labels=[0,1,2,3,4,5]
Colors=['red','orange','yellow','green','blue','purple']
label_express=['anger', 'disgust','fear','happy','sad','surprise'] #0-5对应的标签含义

#二维散点图
#初始化画布
plt.figure(figsize=(8, 6), dpi=80) # figsize定义画布大小，dpi定义画布分辨率
plt.title('Transformed samples via sklearn.decomposition.PCA')
# plt.title('Transformed samples via sklearn.decomposition.LDA')
#分别确定x和y轴的含义及范围
plt.xlabel('x_values')
plt.ylabel('y_values')
plt.xlim([-60,60])
plt.ylim([-40,60])

# 为了不同标签的数据显示不同颜色 不能对读取数据list直接进行输入
# 需要每一个点进行label判断 根据对应数字进行绘制
# 若label情况较少 可采取包含list直接输入的([:,0]/[:,1])其他方法
for tlabel in labels:
    # pca读取数据
    x_pca_data=X_pca[label==tlabel,0]
    y_pca_data=X_pca[label==tlabel,1]
    plt.scatter(x=x_pca_data,y=y_pca_data,s=20,c=Colors[tlabel],label=label_express[tlabel])
    # lda读取数据
    x_lda_data=X_lda[label==tlabel,0]
    y_lda_data=X_lda[label==tlabel,1]
    plt.scatter(x=x_lda_data,y=y_lda_data,s=20,c=Colors[tlabel],label=label_express[tlabel])
plt.legend(loc="upper right") #输出标签信息在右上角
plt.grid()
plt.show()

# #三维散点图
#初始化画布
fig=plt.figure(figsize=(8, 6), dpi=80) # figsize定义画布大小，dpi定义画布分辨率
ax =fig.add_subplot(111,projection='3d')
ax.set_title('Transformed samples via sklearn.decomposition.PCA')
ax.set_title('Transformed samples via sklearn.decomposition.LDA')
#分别确定x和y轴的含义及范围
ax.set_xlabel('x_value')
ax.set_ylabel('y_value')
ax.set_zlabel('z_value')
for tlabel in labels:
    #pca读取数据
    x_pca_data=X_pca[label==tlabel,0]
    y_pca_data=X_pca[label==tlabel,1]
    z_pca_data=X_pca[label==tlabel,2]
    ax.scatter(xs=x_pca_data,ys=y_pca_data,zs=z_pca_data,s=20,c=Colors[tlabel],label=label_express[tlabel])
    # lda读取数据
    x_lda_data=X_lda[label==tlabel,0]
    y_lda_data=X_lda[label==tlabel,1]
    z_lda_data=X_lda[label==tlabel,2]
    ax.scatter(xs=x_lda_data,ys=y_lda_data,zs=z_lda_data,s=20,c=Colors[tlabel],label=label_express[tlabel])
plt.legend(loc="upper right") #输出标签信息在右上角
plt.show()