from math import *
from numpy import *
import random
import datasets
import matplotlib.pyplot as plt

waitForEnter = False


def generateUniformExample(numDim):
    return [random.random() for d in range(numDim)]


def generateUniformDataset(numDim, numEx):
    return [generateUniformExample(numDim) for n in range(numEx)]


def computeExampleDistance(x1, x2):
    dist = 0.0
    for d in range(len(x1)):
        dist += (x1[d] - x2[d]) * (x1[d] - x2[d])
    return sqrt(dist)


def computeDistances(data):
    N = len(data)
    D = len(data[0])
    dist = []
    for n in range(N):
        for m in range(n):
            dist.append(computeExampleDistance(data[n], data[m]) / sqrt(D))
    return dist


def computeExampleDistanceSubdims(x1, x2, subdims):
    dist = 0.0
    for d in subdims:
        dist += (x1[d] - x2[d]) * (x1[d] - x2[d])
    return sqrt(dist)


def computeDistancesSubdims(data, d):
    N = len(data)
    D = len(data[0])
    subdims = random.sample(range(D), d)
    dist = []
    for n in range(N):
        for m in range(n):
            dist.append(computeExampleDistanceSubdims(data[n], data[m], subdims) / sqrt(d))
    return dist


Dims = [2, 8, 32, 128, 512]  # dimensionalities to try
Cols = ['#FF0000', '#880000', '#000000', '#000088', '#0000FF']
Bins = arange(0, 1, 0.02)

plt.xlabel('distance / sqrt(dimensionality)')
plt.ylabel('# of pairs of points at that distance')
plt.title('dimensionality versus uniform point distances')

# distances = computeDistances(datasets.DigitData.Xall.tolist())
# print("D=%d, average distance=%g" % (784, mean(distances) * sqrt(784)))
# plt.hist(distances,
#          Bins,
#          histtype='step',
#          color=Cols[0])
# plt.legend(['%d dims' % 784])

for i, d in enumerate(Dims):
    distances = computeDistancesSubdims(datasets.DigitData.Xall.tolist(), d)
    print("D=%d, average distance=%g" % (d, mean(distances) * sqrt(d)))
    plt.hist(distances,
             Bins,
             histtype='step',
             color=Cols[i])
plt.legend(['%d dims' % d for d in Dims])

plt.savefig('fig.pdf')
plt.show()
