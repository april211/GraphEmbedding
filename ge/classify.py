from __future__ import print_function

import numpy
from sklearn.metrics import f1_score, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer


class TopKRanker(OneVsRestClassifier):
    """
    The code is primarily used for multi-label classification tasks where each input sample can belong to multiple classes, and it ranks the top-k predicted labels for each input sample based on their probability scores.
    """
    def predict(self, X, top_k_list):
        
        # 获得预测概率值（这一步之前模型已经经过了训练）
        probs = numpy.asarray(super(TopKRanker, self).predict_proba(X))

        all_labels = []

        for i, k in enumerate(top_k_list):
            
            # 获取模型预测概率值最大的 k 个类别
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            
            # one-hot 化，完成图节点的多标签预测
            probs_[:] = 0
            probs_[labels] = 1
            all_labels.append(probs_)
        
        return numpy.asarray(all_labels)

class Classifier(object):

    def __init__(self, embeddings, clf):

        self.embeddings = embeddings                    # 用于训练 self.clf 的嵌入向量
        self.clf = TopKRanker(clf)

        # output binary array is desired in CSR sparse format
        self.binarizer = MultiLabelBinarizer(sparse_output=True)

    def train(self, X, Y, Y_all):

        self.binarizer.fit(Y_all)                       # 多标签生成器必须纳入所有的数据类别
        X_train = [self.embeddings[x] for x in X]       # 使用 word2vec 训练得到的 embedding 向量作为训练数据
        Y = self.binarizer.transform(Y)                 # 使用上面 fit 得到的类别信息，对训练数据标签进行处理
        self.clf.fit(X_train, Y)                        # 考虑到多标签分类，Y 是一个标签矩阵

    def split_train_evaluate(self, X, Y, train_precent, seed=0):

        # 储存 initial state
        state = numpy.random.get_state()

        numpy.random.seed(seed)
        shuffle_indices = numpy.random.permutation(numpy.arange(len(X)))
        
        # 确定训练集、测试集的大小
        training_size = int(train_precent * len(X))

        # 按照随机的 indices 分割数据集，分别获得训练集、测试集
        X_train = [X[shuffle_indices[i]] for i in range(training_size)]
        Y_train = [Y[shuffle_indices[i]] for i in range(training_size)]

        X_test = [X[shuffle_indices[i]] for i in range(training_size, len(X))]
        Y_test = [Y[shuffle_indices[i]] for i in range(training_size, len(X))]

        self.train(X_train, Y_train, Y)

        # 恢复 initial state
        numpy.random.set_state(state)

        return self.evaluate(X_test, Y_test)

    def evaluate(self, X, Y):
        """
        针对多标签分类的评估
        """

        top_k_list = [len(l) for l in Y]            # 获取标签的个数，对应多标签分类时的处理
        Y_ = self.predict(X, top_k_list)
        Y = self.binarizer.transform(Y)

        averages = ["micro", "macro", "samples", "weighted"]
        results = {}
        for average in averages:
            results[average] = f1_score(Y, Y_, average=average)
        results['acc'] = accuracy_score(Y, Y_)

        print('-------------------')
        print(results)

        return results

    def predict(self, X, top_k_list):
        X_ = numpy.asarray([self.embeddings[x] for x in X])
        Y = self.clf.predict(X_, top_k_list=top_k_list)
        return Y

def read_node_label(filename, skip_head=False):

    fin = open(filename, 'r')

    X = []
    Y = []

    while True:
        if skip_head:
            fin.readline()
        
        l = fin.readline()
        if l == '':
            break
        
        # 文本数据使用空格分割
        vec = l.strip().split(' ')
        X.append(vec[0])
        Y.append(vec[1:])           # 多标签分类时，单个对象可能有多个类别标签
    
    fin.close()

    return X, Y

