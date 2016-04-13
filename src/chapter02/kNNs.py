# -*- coding:utf-8 -*-

'''
Created on 2016.04.12
@author: YangXian
'''

'''
kNN算法总结：

    优点：
    1. 精度高，对异常值不敏感，无数据输入假定？
    
    缺点：
    1. 空间计算复杂度高，计算量比较大，比如本代码中的数字识别
    2. 无法给出数据的内在含义，而真的只是简答你的数据
    
    因此：
    K-邻近算法使用的场合：计算复杂度不那么大；数据本身数以数值型或者标称型

'''


from numpy import array,tile,zeros    # 导入numpy包
import operator                       # 导入字符操作包
from os import listdir                # 可以列出指定目录的文件名
import time
# import matplotlib.pyplot as plt
# from matplotlib.pyplot import title,xlabel,ylabel
from matplotlib.font_manager import FontProperties
from chapter01.practice import start

font=FontProperties(fname=r"c:\windows\fonts\simsun.ttc",size=12);

# 定义数据创建函数
def createDataSet():
    group=array([[1.0,1.1],[1.0,1.0],[0.0,0.0],[0.0,0.1]]);
    labels=['A','A','B','B'];
    return group,labels;


def classify0(inX,dataSet,labels,k):
    dataSetSize=dataSet.shape[0];                 # 取数据集的行数
#     print '样本数据集的个数为： ',dataSetSize,'\n'
    diffMat=tile(inX, (dataSetSize,1))-dataSet;   # 将测试数据纵向扩充为样本集的个数好进行距离计算
    sqDiffMat=diffMat**2;                         # 计算样本与测试数据的欧式距离各分量；
    sqDistances=sqDiffMat.sum(axis=1);            # 计算欧式距离的平方
    distances=sqDistances**0.5;
#     print '与样本集合的欧式距离值为：',distances,'\n';
    sortedDistIndicies=distances.argsort();                        # 返回数组从小到达的索引值，依次是，数组中最小数的索引，。。。
    
    classCount={};
    for i in range(k):
        voteILabel=labels[sortedDistIndicies[i]];                  # 找出第i近的样本数据的标签
        classCount[voteILabel]=classCount.get(voteILabel,0)+1;     # 为标签赋值，当标签不存在时返回0；
#     print classCount
    sortedClassCount=sorted(classCount.iteritems(),key=operator.itemgetter(1), reverse=True);   #   将字典第二个元素排序
#     print '============================'
#     print sortedClassCount,'\n'
    return sortedClassCount[0][0];         # 返回第0个元祖的第0个元素，即label



# ==================================================我是分割线============================================================
'''
   本函数使用K临近算法改善约会网站的配对效果
   @author:  YangXian
   @copyright: Copyright  By YangXian
   Created on 2016.04.12
'''

def file2matrix(filename): 
    fr=open(filename);                     # 打开文件，获得文件对象；
    arrayLines=fr.readlines();             # 按行读取文件内容；
    numOfLines=len(arrayLines);            # 计算文本行数
    returnMat=zeros((numOfLines,3));       # 生成与文本行数相同，列为3的零矩阵；
    classLabelVector=[];                   # 创建分类标签空向量；
    index=0;
    
    for line in arrayLines:
        line=line.strip();                 # 用于移除字符串头尾指定的字符（默认为空格），这里是去掉回车字符
        listFromLine=line.split('\t');     # 通过指定分隔符对字符串进行切片，如果参数num有指定值，则仅分隔 num个子字符串，使用tab将整行数据分割成一个元素列表
        returnMat[index,:]=listFromLine[0:3];                 # 返回分割后的前3个子字符串，3在Python取不到；
        classLabelVector.append(int(listFromLine[-1]));       # 返回分割后的最后一列的值，并转化为整数；
        index+=1;
    return returnMat,classLabelVector;



# 主函数执行部分
'''
group,labels=createDataSet();
print '测试集属于%s类！' %classify0([0.0,0.0], group, labels, 2);

'''

# datingDataMat,datingLabels=file2matrix('textdata/datingTestSet2.txt');
# print datingDataMat;
# print datingLabels[0:20];

# fig=plt.figure();               # 创建图标
# ax=fig.add_subplot(211);        # 添加子图
# ax.scatter(datingDataMat[:,1],datingDataMat[:,2]);       # 绘制玩游戏(横)时间比例与每周消费冰淇淋公升数散点图

# ax.scatter(datingDataMat[:,1],datingDataMat[:,2],20.0*array(datingLabels),20.0*array(datingLabels));    # 系数表示点的大小
# xlabel(u'玩视频游戏所占时间百比',fontproperties=font)
# ylabel(u'每周吃冰淇淋',fontproperties=font);
# title(u'吃冰淇淋数跟玩游戏时间之间的关系',fontproperties=font);

# ax.scatter(datingDataMat[:,0],datingDataMat[:,1],20.0*array(datingLabels),20.0*array(datingLabels));    # 系数表示点的大小
# xlabel(u'每年获取的飞行常客里程数',fontproperties=font)
# ylabel(u'玩视频游戏时间百分比数',fontproperties=font);
# title(u'游戏时间-飞行里程',fontproperties=font);



# 数据归一化处理
def autoNorm(dataSet):
    minVals=dataSet.min(0);                  # 0表示从列选取最小值，是一个参数，而不是索引；索引需要用[]括号；
    maxVals=dataSet.max(0);
    ranges=maxVals-minVals;
    #normDataSet=zeros(shape(dataSet));      # 这一行貌似是多余的
    m=dataSet.shape[0];
    normDataSet=dataSet-tile(minVals,(m,1));
    normDataSet=normDataSet/tile(ranges,(m,1));
    return normDataSet,ranges,minVals

# normMat,ranges,minVals=autoNorm(datingDataMat);

# ax2=fig.add_subplot(212);         # 添加子图
# ax2.scatter(normMat[:,0],normMat[:,1],20.0*array(datingLabels),20.0*array(datingLabels));    # 系数表示点的大小
# xlabel(u'每年获取的飞行常客里程数',fontproperties=font)
# ylabel(u'玩视频游戏时间百分比数',fontproperties=font);
# title(u'游戏时间-飞行里程',fontproperties=font);
# plt.show()


def datingClassTest():
    hoRatio = 0.50      #hold out 10%
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')       #load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print "the total error rate is: %f" % (errorCount/float(numTestVecs))
    print errorCount



def classifyPersion():
    resultList=['not at all','in small doses','in large doses'];                        # 定义结果列表
    percentTats=float(raw_input("Percentage of time spent on playing video games?"));
    ffMiles=float(raw_input("Frequent flier miles earned per year?"));                  # input会将用户输入来做类型转换，而raw_input则作为字符串来处理；
    iceCream=float(raw_input("Liters of ice cream consumed per year?"));                # input会将用户输入来做类型转换，而raw_input则作为字符串来处理；
    datingDataMat,datingLabels=file2matrix('textdata/datingTestSet2.txt');
    normMat,ranges,minVals=autoNorm(datingDataMat);
    inArr=array([ffMiles,percentTats,iceCream]);
    classifierResult=classify0(inArr-minVals, normMat, datingLabels, 3);
    print "You will probably like this person: ",resultList[classifierResult-1];

#  ========================================================手写识别系统====================================================================

# 将数组转化为向量以便后续使用
def img2vector(filename):
    returnVect=zeros((1,1024));           # 参数只有一个，将整个元祖作为参数，而不是由1,1024两个参数
    fr=open(filename);
    for i in range(32):
        lineStr=fr.readline();            # readline和readlines的差别
        for j in range(32):
            returnVect[0,32*i+j]=int(lineStr);
    return returnVect;

def handWrintingClassTest():
    hwLabels=[];            # 分类标签
    trainingFileList=listdir('textdata/trainingDigits');     # 列出目录下的所有文件，得到文件名列表
    m=len(trainingFileList);                                 # 求文件的个数
    trainingMat=zeros((m,1024));                             # 生成文件数量行，1024列的零矩阵
    for i in range(m):
        fileNameStr=trainingFileList[i];
        fileStr=fileNameStr.split('.')[0];                   # 根据文件名去掉.号后的第一个字符，也就是后缀
        classNumStr=int(fileStr.split('_')[0]);              # 得到文件实际属于的类标签
        hwLabels.append(classNumStr);
        trainingMat[i,:]=img2vector('textdata/trainingDigits/%s'%fileNameStr);
    
    testFileList=listdir('textdata/testDigits');             # 列出测试目录下所有的文件
    errorCount=0.0;                                          # 默认错误率为0
    mTest=len(testFileList);
    for i in range(mTest):
        fileNameStr=testFileList[i];
        fileStr=fileNameStr.split('.')[0];
        classNumStr=int(fileStr.split('_')[0]);
        vectorUnderTest=img2vector('textdata/testDigits/%s'%fileNameStr);
        classifierResult=classify0(vectorUnderTest, trainingMat, hwLabels, 3);           # 调用已经写好的分类器，进行测试，监督学习无需训练
#         print 'The classifier came back with %d, the real answer is :%d'%(classifierResult,classNumStr);
        if(classifierResult != classNumStr):errorCount+=1;
    
    print "\nThe total number of errors is:%d "%errorCount;
    print "The total error rate is :%f"%(errorCount/float(mTest));
    
#  ========================================================主函数====================================================================
if __name__ =="__main__":
#     classifyPersion();
    start=time.clock();
    handWrintingClassTest();
    end=time.clock();
    print "程序总的执行时间是：%f"%-(start-end)
    




