import numpy as np

from ge.classify import read_node_label, Classifier
from ge import DeepWalk
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE

"""
DeepWalk算法主要包括两个步骤，第一步为随机游走采样节点序列，第二步为使用 Skip-Gram model 学习表达向量。

① 构建同构网络，从网络中的每个节点开始分别进行 Random Walk 采样，得到局部相关联的训练数据； 
② 对采样数据进行 Skip-Gram 训练，将离散的网络节点表示成向量化，并最大化节点共现；最后使用 Hierarchical Softmax 来做超大规模分类的分类器。
"""

def evaluate_embeddings(embeddings):

    # 读入节点编号和类别标签
    X, Y = read_node_label('./data/wiki/wiki_labels.txt')

    tr_frac = 0.8
    print("Training classifier using {:.2f}% nodes...".format(tr_frac * 100))
    
    # 使用 embeddings 对图节点进行分类
    clf = Classifier(embeddings=embeddings, clf=LogisticRegression())
    clf.split_train_evaluate(X, Y, tr_frac)


def plot_embeddings(embeddings,):
    X, Y = read_node_label('./data/wiki/wiki_labels.txt')

    emb_list = []
    for k in X:
        emb_list.append(embeddings[k])
    emb_list = np.array(emb_list)

    # 执行数据降维，进行可视化
    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)

    color_idx = {}
    for i in range(len(X)):
        color_idx.setdefault(Y[i][0], [])           # 多标签仅展示第一个类别
        color_idx[Y[i][0]].append(i)

    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)
    plt.legend()
    plt.show()


if __name__ == "__main__":

    # 通过输入边来构造图数据结构
    G = nx.read_edgelist('./data/wiki/Wiki_edgelist.txt',
                         create_using=nx.DiGraph(), 
                         nodetype=None, 
                         data=[('weight', int)]
        )

    # 生成图嵌入向量
    model = DeepWalk(G, walk_length=10, num_walks=80, workers=1)      # TODO **这里开多线程会报 import 错误**
    model.train(window_size=5, iter=3, workers=3)                     # 这里的参数将会被送入 Word2Vec 
    embeddings = model.get_embeddings()

    # 将 embeddings 用于节点分类任务，并对结果进行评估、可视化
    evaluate_embeddings(embeddings)
    plot_embeddings(embeddings)

"""
待解决问题：

1. `ImportError: cannot import name 'LSTM' from 'tensorflow.python.keras.layers'`
    经过观察 `traceback` 日志并调整 `DeepWalk` 的 `workers` 数目，发现问题出现在 `deepctr` 库的
        `from tensorflow.python.keras.layers import LSTM, Lambda, Layer, Dropout`
    导入语句，该错误在 `workers > 1` 时出现，原因未知。

"""

