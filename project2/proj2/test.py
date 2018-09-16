from sklearn.tree import DecisionTreeClassifier
import multiclass
import util
from datasets import *
from timeit import timeit

# h = multiclass.OAA(5, lambda: DecisionTreeClassifier(max_depth=3))
# h.train(WineDataSmall.X, WineDataSmall.Y)
# P = h.predictAll(WineDataSmall.Xte)
# print(mean(P == WineDataSmall.Yte))
# print(mean(WineDataSmall.Yte == 1))
# print(mode(WineDataSmall.Y))
# print(WineDataSmall.labels[1])
# util.showTree(h.f[2], WineDataSmall.words)

# h = multiclass.AVA(5, lambda: DecisionTreeClassifier(max_depth=3))
# h.train(WineDataSmall.X, WineDataSmall.Y)
# P = h.predictAll(WineDataSmall.Xte)
# print(mean(P == WineDataSmall.Yte))
# print(mean(WineDataSmall.Yte == 1))
# print(WineDataSmall.labels[0])
# print()
# util.showTree(h.f[2][0], WineDataSmall.words)


# h = multiclass.OAA(20, lambda: DecisionTreeClassifier(max_depth=3))
# h.train(WineData.X, WineData.Y)
# P = h.predictAll(WineData.Xte)
# print(mean(P == WineData.Yte))
# P = h.predictAll(WineData.Xte, useZeroOne=True)
# print(mean(P == WineData.Yte))


# s = "h = multiclass.OAA(20, lambda: DecisionTreeClassifier(max_depth=3))\n" \
#   + "h.train(WineData.X, WineData.Y)\nP = h.predictAll(WineData.Xte)"
# r = "from sklearn.tree import DecisionTreeClassifier\nimport multiclass, util\nfrom datasets import WineData"
# t = timeit(stmt=s, number=1, setup=r)
# print(t)

# t = multiclass.makeBalancedTree(range(5))
# h = multiclass.MCTree(t, lambda: DecisionTreeClassifier(max_depth=3))
# h.train(WineDataSmall.X, WineDataSmall.Y)
# P = h.predictAll(WineDataSmall.Xte)
# print(mean(P == WineDataSmall.Yte))

t = multiclass.makeBalancedTree(range(20))
h = multiclass.MCTree(t, lambda: DecisionTreeClassifier(max_depth=3))
h.train(WineData.X, WineData.Y)
P = h.predictAll(WineData.Xte)
print(mean(P == WineData.Yte))

# x = array([[ 0,  1,  2],
#            [ 3,  4,  5],
#            [ 6,  7,  8],
#            [ 9, 10, 11]])
#
# rows = np.array([[0, 0],
#                  [3, 3]], dtype=np.intp)
#
# columns = np.array([[0, 2],
#                     [0, 2]], dtype=np.intp)
#
# print(x[rows, columns])
