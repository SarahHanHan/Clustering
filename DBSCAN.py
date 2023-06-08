#DBSCAN是基于密度空间的聚类算法，与KMeans算法不同，它不需要确定聚类的数量，而是基于数据推测聚类的数目，它能够针对任意形状产生聚类。
#DBSCAN算法需要首先确定两个参数：
#（1）epsilon:在一个点周围邻近区域的半径
#（2）minPts:邻近区域内至少包含点的个数
#根据以上两个参数，结合epsilon-neighborhood的特征，可以把样本中的点分成三类：

#核点（core point）：满足NBHD(p,epsilon)>=minPts，则为核样本点
#边缘点（border point）：NBHD(p,epsilon)<minPts，但是该点可由一些核点获得（density-reachable或者directly-reachable）
#离群点（Outlier）：既不是核点也不是边缘点，则是不属于这一类的点

#DBSCAN步骤
#DBSCAN的一般步骤是：（在已知epsilon和minPts的前提下）

#1.任意选择一个点（既没有指定到一个类也没有特定为外围点），计算它的NBHD(p,epsilon)判断是否为核点。如果是，在该点周围建立一个类，否则，设定为外围点。
#2.遍历其他点，直到建立一个类。把directly-reachable的点加入到类中，接着把density-reachable的点也加进来。如果标记为外围的点被加进来，修改状态为边缘点。
#3.重复步骤1和2，直到所有的点满足在类中（核点或边缘点）或者为外围点

#Code by SarahHan
#coding = utf-8
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import pandas as pd

data = pd.read_csv("Wholesale customers data.csv")
data.drop(['Channel', 'Region'], axis = 1, inplace = True)

data = data[['Grocery','Milk']]
#转换成数据 convert to array
data = data.values.astype("float32", copy=False)#convert to array

#数据预处理，特征标准化，每一维是零均值和单位方差
#StandardScaler类是处理数据归一化和标准化
#fit()函数求得训练集X的均值，方差，最大值，最小值,这些训练集X固有的属性
#transform()函数在fit函数的基础上，进行标准化，降维，归一化等操作
#归一化的目的就是使得预处理的数据被限定在一定的范围内（比如[0,1]或者[-1,1]），从而消除奇异样本数据导致的不良影响
#如果不进行归一化，那么由于特征向量中不同特征的取值相差较大，会导致目标函数变“扁”。这样在进行梯度下降的时候，梯度的方向就会偏离最小值的方向，走很多弯路，即训练时间过长
#如果进行归一化以后，目标函数会呈现比较“圆”，这样训练速度大大加快，少走很多弯路
stscaler = StandardScaler().fit(data)
data = stscaler.transform(data)

#画出x和y的散点图
plt.scatter(data[:,0],data[:,1])
plt.xlabel('Groceries')
plt.ylabel('Milk')
plt.title('Wholesale Data - Groceries and Milk')
plt.savefig("wholesale.png", format='PNG')

dbsc = DBSCAN(eps=0.9, min_samples=10).fit(data)

#聚类得到每个点的聚类标签，-1表示噪点
labels = dbsc.labels_
print(labels)
#构造和labels一致的零矩阵，值是False
core_samples = np.zeros_like(labels,dtype=bool)
core_samples[dbsc.core_sample_indices_] = True
print(core_samples)

unique_labels = np.unique(labels)
#linespace返回在【0，1】之间均匀分布数字是len个，Spectral生成len个颜色
colors = plt.cm.Spectral(np.linspace(0,1,len(unique_labels)))
print(unique_labels,colors)

for(label,color) in zip(unique_labels,colors):
    class_member_mask = (labels == label)
    print(class_member_mask & core_samples)
    xy = data[class_member_mask &core_samples]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=color, markersize=10)

    xy2 = data[class_member_mask & ~core_samples]
    plt.plot(xy2[:, 0], xy2[:, 1], 'o', markerfacecolor=color, markersize=5)


plt.title("DBSCAN on Wholsesale data")
plt.xlabel("Grocery(scaled)")
plt.ylabel("Milk(scaled)")
plt.savefig("(0.9,10)dbscan_wholesale.png", format="PNG")

