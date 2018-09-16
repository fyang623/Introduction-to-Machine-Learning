"""
Implementation of k-nearest-neighbor classifier
"""

from pylab import *
from binary import *
import datasets
import runClassifier


class KNN(BinaryClassifier):
    """
    This class defines a nearest neighbor classifier, that support
    _both_ K-nearest neighbors _and_ epsilon ball neighbors.
    """

    def __init__(self, opts):
        """
        Initialize the classifier.  There's actually basically nothing
        to do here since nearest neighbors do not really train.
        """

        # remember the options
        self.opts = opts

        # just call reset
        self.reset()

    def reset(self):
        self.trX = zeros((0, 0))  # where we will store the training examples
        self.trY = zeros(0)  # where we will store the training labels

    def online(self):
        """
        We're not online
        """
        return False

    def __repr__(self):
        """
        Return a string representation of the tree
        """
        return "w=" + repr(self.weights)

    def predict(self, X):
        """
        X is a vector that we're supposed to make a prediction about.
        Our return value should be the 'vote' in favor of a positive
        or negative label.  In particular, if, in our neighbor set,
        there are 5 positive training examples and 2 negative
        examples, we return 5-2=3.

        Everything should be in terms of _Euclidean distance_, NOT
        squared Euclidean distance or anything more exotic.
        """

        isKNN = self.opts['isKNN']  # true for KNN, false for epsilon balls
        N = self.trX.shape[0]  # number of training examples

        if self.trY.size == 0:
            return 0  # if we haven't trained yet, return 0

        if isKNN:
            # this is a K nearest neighbor model
            # hint: look at the 'argsort' function in numpy
            K = min(self.opts['K'], N)  # how many NN to use
            ## TODO: YOUR CODE HERE
            # The implementation below occupies a lot of memory if the training data is large.
            distances = [square(self.trX[i, :] - X).sum() for i in range(N)]
            sortedDistIndices = argsort(array(distances))
            return sum([self.trY[i] for i in sortedDistIndices[:K]])

            # Below is an alternative implementation, which is memory-efficient.
            #
            # neighbors = [*range(K)]
            # distances = [square(self.trX[i, :] - X).sum() for i in range(K)]
            # threshDistIndex = argsort(array(distances))[K-1]
            # threshDist= distances[threshDistIndex]
            # for i in range(K, N):
            #     d = square(self.trX[i, :] - X).sum()
            #     if d == threshDist:
            #         neighbors.append(i)
            #         distances.append(d)
            #     elif d < threshDist:
            #         neighbors.append(i)
            #         distances.append(d)
            #         sortedDistIndices = argsort(array(distances))
            #         if distances[sortedDistIndices[K-1]] < distances[sortedDistIndices[K]]:
            #             newNeighbors = []
            #             newDistances = []
            #             for i in sortedDistIndices[:K]:
            #                 newNeighbors.append(neighbors[i])
            #                 newDistances.append(distances[i])
            #             neighbors = newNeighbors
            #             distances = newDistances
            #         threshDistIndex = argsort(array(distances))[K-1]
            #         threshDist = distances[threshDistIndex]
            # return sum([self.trY[i] for i in [neighbors[i] for i in argsort(array(distances))[:K]]])

        else:
            # this is an epsilon ball model
            eps = self.opts['eps']  # how big is our epsilon ball

            # val = 0  # this is our return value: #pos - #neg within an epsilon ball of X
            ### TODO: YOUR CODE HERE
            neighbors = []
            for i in range(N):
                d = square(self.trX[i, :] - X).sum()
                if d <= eps ** 2:
                    neighbors.append(i)
            return sum([self.trY[i] for i in neighbors])

    def getRepresentation(self):
        """
        Return the weights
        """
        return (self.trX, self.trY)

    def train(self, X, Y):
        """
        Just store the data.
        """
        self.trX = X
        self.trY = Y


# for i in [0.5, 1.0, 2.0]:
#     runClassifier.trainTestSet(KNN({'isKNN': False, 'eps': i}), datasets.TennisData)

# for i in range(1, 6, 2):
#     runClassifier.trainTestSet(KNN({'isKNN': True, 'K': i}), datasets.TennisData)

# for i in range(6, 11, 2):
#     runClassifier.trainTestSet(KNN({'isKNN': False, 'eps': i}), datasets.DigitData)

# for i in range(1, 6, 2):
#     runClassifier.trainTestSet(KNN({'isKNN': True, 'K': i}), datasets.DigitData)

# for i in range(7, 11):
#     curve = runClassifier.learningCurveSet(KNN({'isKNN': False, 'eps': i}), datasets.DigitData)
#     runClassifier.plotCurve('Epsilon-Ball NN on Digit Data (eps = {})'.format(i), curve)
#
# for i in range(1, 6):
#     curve = runClassifier.learningCurveSet(KNN({'isKNN': True, 'K': i}), datasets.DigitData)
#     runClassifier.plotCurve('K-NN on Digit Data (K = {})'.format(i), curve)
