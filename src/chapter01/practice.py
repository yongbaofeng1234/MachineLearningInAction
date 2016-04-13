#_*_coding:utf-8_*_

from numpy import *
import time

start=time.time()

a=random.rand(4,4)
print a
print "\n"
randMat=mat(a)
print randMat
print "\n"
print randMat.I

end=time.time()

print "\n"
print randMat*randMat.I       #存在数值的误差
print "\n"

print eye(4)

print "\n"
print 'Running time: ', end-start,"s"