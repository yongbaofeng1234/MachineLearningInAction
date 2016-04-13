# -*- coding:utf-8 -*-

'''
Created on 2016.04.13
@author: YangXian
'''
'''
   朴素贝叶斯分类方法基本知识：
   
        朴素也贝斯的基本假设： 1. 假设特征之间相互独立，即某个特征的出现跟另外的木有关系；  
                  2. 假设特征都具有同等地位；
        
   
   算法关键点：
   
      优点：
    1. 在训练数据较少的情况下仍然可用；    2. 可以处理多类别问题
    
    缺点：
    1. 对于输入数据的准备方式比较敏感
    
    因此：
         算法使用的场合：   标称值数据类型
'''

# 创建文本集合
from numpy import *
import time

def loaddataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']];
    classVec=[0,1,0,1,0,1];                  # 1代表侮辱性文字，0代表正常言论
    return postingList,classVec;   

# 创建一个包含在所有文档中出现单不重复的文档词汇列表
def createVocabList(dataSet):
    vocabSet=set([]);
    for document in dataSet:
        vocabSet=vocabSet | set(document);   # 创建两个集合的并集
    return list(vocabSet);                   # 转化为列表


# 计算某个文档在词汇列表中的文档向量，向量大象就是文档不重复词汇数
def setOfWords2Vec(vocabList,inputSet):
    returnVec=[0]*len(vocabList);                     # 创建大小与文档词汇表相同的0向量；
    for word in inputSet:                             # 
        if word in vocabList:
            returnVec[vocabList.index(word)]=1;       # 将文档出现在词汇在词汇列表中标注为1
        else: print "The word: %s is not in my Vocabulary!" %word
    return returnVec;

# 朴素贝叶斯分类器训练函数
def trainNB0(trainMatrix,trainCategory):              # trainCategory是分类标签向量，即训练文档所属的分类向量
    numTrainDocs=len(trainMatrix);                    # 训练文档个数，这里的矩阵行是某一个文档被向量话后的setOfWords2Vec
    numWords=len(trainMatrix[0]);                     # 计算词汇表词汇数量；
    pAbusive=sum(trainCategory)/float(numTrainDocs);  # 文档标签数即侮辱性还是非侮辱性占整个训练文档数量的比例；  
    p0Num=ones(numWords);
    p1Num=ones(numWords);                            # 初始化分母和分子        
    p0Denom=2.0;
    p1Denom=2.0;
    for i in range(numTrainDocs):                     # 遍历每个文档
        if trainCategory[i]==1:                       # 如果属于侮辱性文档
            p1Num +=trainMatrix[i];                   # 求侮辱性文档的一个词类集合向量
            p1Denom +=sum(trainMatrix[i]);            # 侮辱性词语的一个数量集合，这里可能出现重复计算的情况
        else:
            p0Num +=trainMatrix[i];
            p0Denom +=sum(trainMatrix[i]);
    p1Vect=log(p1Num/p1Denom);             # 计算词属于侮辱性文档的概率
    p0Vect=log(p0Num/p0Denom);             # 计算词不属于侮辱性文档的概率
    return p0Vect,p1Vect,pAbusive;

# 正式构建分类器
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):     # 使用训练好的向量
    p1=sum(vec2Classify*p1Vec)+log(pClass1); 
    p0=sum(vec2Classify*p0Vec)+log(1.0-pClass1); 
    if p1>p0:
        return 1;
    else:
        return 0;


 
 
 
 
# ============================================以下是主函数部分了=================================================            
if __name__=='__main__':
    start=time.clock();
    listOPosts,listClasses = loaddataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)
    end=time.clock();
    print "程序总的执行时间是：%f秒"%-(start-end)






















