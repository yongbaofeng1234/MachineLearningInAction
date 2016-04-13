# -*- coding:utf-8 -*-

'''
Created on 2016.04.12

@author: YangXian 
'''

'''
决策树算法总结：


     算法关键点：如何确定每次分类属性的中作用最大的属性；
     方法：1. 使用信心论来度量数据集的信息；    2. 计算如果采用某种属性分类的分类前后信息的一个增益值；     3. 选择信息增益最大的属性作为当前起决定性作用的分类属性进行分类
     
     信息：l(xi)=-log2(p(xi));   即符号xi的信息为：符号xi选择所有可能分类中某一类的概率
     
    优点：
    1. 精度高，对异常值不敏感；           2. 能够直接明了知道数据的真实含义；             3. 计算量也不大；                  4. 可以使用不熟悉的数据集合，从中提取出系列规则
    
    缺点：
    1. 可能会产生过度匹配问题
    2. 虽然测试数据集很快，但是构建决策树却很费时；
    
    因此：
         决策树算法使用的场合：    计算复杂度不那么大；     数据本身数以数值型或者标称型

'''

from math import log
import operator


# 计算一个给定数据集的香浓熵
def calcShannonEnt(dataSet):
    numEntries=len(dataSet);                              # 计算数据集的项数，或者说广义的长度，实例总数；
    labelCount={};                                        # 创建所有可能分类的字典
    for featVec in dataSet:
        currentLabel=featVec[-1];                         # 去实例最后一列的值为属性； 
        if currentLabel not in labelCount.keys():         # 如果改属性在字典里不存在，则添加字典并将值设为0；
            labelCount[currentLabel]=0;
        labelCount[currentLabel]+=1;                      # 如果存在，则将键的值增加数量；
    
    shannonEnt=0.0;
    for key in labelCount:
        prob=float(labelCount[key])/numEntries;           # 某个属性在整个数据集中的出现概率；
        shannonEnt -= prob*log(prob,2);                   # 总的信息熵，也即是所有分类的信息之和 
    return shannonEnt;
        

# 创建数据集
def createDataSet():
    dataSet=[[1,1,'yes'],
             [1,1,'yes'],
             [1,1,'no'],
             [0,1,'no'],
             [0,1,'no']];
    labels=['no surfacing','flippers'];
    return dataSet,labels;         

# 按照给定特征划分数据集
def splitDataSet(dataSet,axis,value):               # 3个参数，数据集，划分特征和需要返回的特征值
    retDataSet=[];
    for featVec in dataSet:                         #     
        if(featVec[axis]==value):                   # 发现数据实例的特征等于某个值，则返回该实例，并抽取改属性外的所有属性
            reducedFeatVec=featVec[:axis];
            reducedFeatVec.extend(featVec[axis:]);
            retDataSet.append(reducedFeatVec);
    return retDataSet;                              # 返回满足属性axis=value的所有实例
            
def chooseBestFeatureToSplit(dataSet):
    numFeatures=len(dataSet[0])-1;          # 数据集的第0行的长度-1，记录特征属性的个数,减去最后一行；
    baseEntropy=calcShannonEnt(dataSet);    # 基准熵，即远数据的上，也就是按最后一个属性分类的熵
    bestInfoGain=0.0;
    bestFeature=-1;                         # 初始化最有特征和对应的信息增益
    
    for i in range(numFeatures):            # 外层循环，对每个属性分类方法进行测试
        featList=[example[i] for example in dataSet];
        uniqueVals=set(featList);           # set函数返回不同字符串组成的集合，重复的计数一次，这里用来计数第i个属性所有可能取值
        newEntropy=0.0;
        for value in uniqueVals:
            subDataSet=splitDataSet(dataSet, i, value);    # 计算第i个属性值等于value时的信息熵，遍历得到所有可能value的熵的总和
            prob=len(subDataSet)/float(len(dataSet));
            newEntropy +=prob+calcShannonEnt(subDataSet);
        infoGain=baseEntropy-newEntropy;                   # 信息增益情况，用基准熵减去按该类分类数据集的熵；
        if(infoGain>bestInfoGain):
            bestInfoGain=infoGain;
            bestFeature=i;
    return bestFeature;                                    # 返回最好的分类属性的序号值

# 计算某个列表中某个属性值出现次数最多的属性，用于在遍历完所有属性，单符号属性仍然不完全相同时，采用的办法
def majorityCont(classList):
    classCount={};
    for vote in classList:
        if vote not in classCount.keys():classCount[vote]=0
        classCount[vote] +=1;
    sortedClassCount=sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True);
    return sortedClassCount[0][0];                         #  key=operator.itemgetter(1)表示按list中的vote元祖的第二个元素进行排序，且是倒叙；

# 利用递归创建决策树
def createTree(dataSet,labels):
    classList=[example[-1] for example in dataSet];        # 基属性的所有取值list
    if classList.count(classList[0])==len(classList):      # 第一个属性值得个数，与总数相等时，则停止划分
        return classList[0];
    if len(dataSet[0])==1:
        return majorityCont(classList);                    # 如果遍历完所有特征，则返回出现次数最多的；
    bestFeature=chooseBestFeatureToSplit(dataSet);
    bestFeatLabel=labels[bestFeature];
    myTree={bestFeatLabel:{}};
    del(labels[bestFeature]);
    featValues=[example[bestFeature] for example in dataSet];          # 重新构造数据集合标签集
    uniqueValues=set(featValues);
    for value in uniqueValues:
        subLabels=labels[:];                                           # 取遍历剩下的所有属性标签值
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet, bestFeature, value), subLabels);
    return myTree;
    

# =====================================================完整决策树分类器构造完成========================================================

# 使用构建的决策树进行分类，构造分类器
def classify(inputTree,featLabels,testVec):       # 3个参数，第二个是属性列表，第三个是测试数据
    firstStr=inputTree.keys()[0];                 # 取出树种的第一个分类特征
    secondDict=inputTree[firstStr];               # 提取第一个分类属性的字典key=firstStr的value值
    featIndex=featLabels.index(firstStr);         # 寻找firstStr属性在featLabels中的Index
    for key in secondDict.keys():                 # 实现递归分类
        if testVec[featIndex]==key:
            if type(secondDict[key]).__name__=='dict':
                classLabel=classify(secondDict[key], featLabels, testVec);
            else: classLabel=secondDict[key];
    return classLabel;
    





