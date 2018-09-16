import numpy as np
import scipy.sparse
import scipy.optimize


class SoftmaxRegression:
    """
    Here you will fill in this incomplete implementation of Softmax regression.

    Adapted from code by Jatin Shah
    """

    def __init__(self, numClasses, exSize, opts={'maxIter': 400}):
        """
        numClasses:     number of possible classifications
        exSize:         size of attribute array (number of input features)
        reg:            regularizing term coefficient (lambda)
        opts:           in this class the only option used is maxIter
        """

        self.numClasses = numClasses
        self.exSize = exSize
        self.opts = opts
        self.indicator = None

        # Initialize weight matrix with empty matrix
        self.W = np.zeros((numClasses, exSize))

    def reset(self, numClasses, exSize, opts={'maxIter': 400}):

        self.__init__(numClasses, exSize, opts)

    def setOption(self, optName, optVal):
        """
        optName:        name of option
        optVal:         new value to assign option to
        """

        self.opts[optName] = optVal

    def cost(self, X, Y, W=None):
        """
        Calculate the cost function for X and Y using current weight matrix W. Note that we are not using
        a regularizer in the cost; this is equivalent to lambda = 0.

        X:              (M x N) matrix of input feature values,
                            where M = exSize, N = number of examples
        Y:              (N x 1) array of expected output classes for each example

        Returns the cost and its gradient, which is the form needed to use scipy.optimize.minimize
        """

        numClasses = self.numClasses
        exSize = self.exSize
        indicator = self.indicator

        W = (W.reshape(numClasses, exSize), self.W)[W is None]  # Ensure W is in the correct dimensions

        # W_X is our activation matrix with dimensions (A * B) where A is the number of classes and B
        # is the number of examples. (W_X[a, b] gives the activation of example b for class a.) You will
        # use this matrix to find the probabilities that example b is class a using the softmax formula.
        W_X = W.dot(X)

        # W_X = W_X - np.max(W_X)
        W_X = W_X - W_X.max(axis=0)

        # TODO: Compute the predicted probabilities, the total cost, and the gradient.

        # Each column of W_X is the set of activations for each class corresponding to 
        # one example; the probabilities are given by the exponential of each entry
        # divided by the sum of the exponentials over the entire column.

        # The cost associated with a single example is given by -1 times the log probability
        # of the true class; initialize the cost variable to the AVERAGE cost over all the examples.
        # Hint: there's an easy way to do this with the indicator matrix.

        # The gradient has the same dimensions as W, and each component (i,j) represents the
        # derivative of the cost with respect to the weight associated with class i, attribute j.
        # The gradient associated with a single example x is given by -1 * A * x_T, where x_T is
        # the transpose of the example, and A is a vector with component i given by (1 - P(class = i))
        # if the true class is i, and (-P(class = i)) otherwise. Notice that this multiplication gives
        # the desired dimensions. Find the AVERAGE gradient over all the examples. Again, there is
        # an easy way to do this with the indicator matrix.

        ### YOUR CODE HERE ###
        probabilities = np.exp(W_X)
        probabilities = probabilities / probabilities.sum(axis=0)
        cost = np.average(np.log((indicator * probabilities).sum(axis=0))) * (-1)
        indicator = indicator - probabilities
        gradient = - np.matmul(indicator, X.T) / len(Y)
        ### YOUR CODE HERE ###

        # flatten is needed by scipy.optimize.minimize
        return cost, gradient.flatten()

    def train(self, X, Y):
        """
        Train to find optimal weight matrix W. Here we make use of the SciPy optimization library but
        in theory you could implement gradient descent to do this as well.
        
        X:              (M x N) matrix of input feature values,
                            where M = exSize, N = number of examples
        Y:              (N x 1) array of expected output classes for each example
        maxIter:        Maximum training iterations
        """

        # This is the indicator function used in the loss function, where indicator[a, b] = 1
        # when example b is labeled a (according to the target Y) and indicator[a, b] = 0 otherwise.
        self.indicator = scipy.sparse.csr_matrix((np.ones(len(Y)), (Y, np.array(range(len(Y))))))
        self.indicator = np.resize(np.array(self.indicator.todense()), (self.numClasses, len(Y)))

        # Set maxIter hyperparameter
        if self.opts['maxIter'] is None:
            self.opts['maxIter'] = 400

        # Lambda function needed by scipy.optimize.minimize
        J = lambda w: self.cost(X, Y, w)

        # SciPy is a powerful data science library, check it out if you're interested :)
        result = scipy.optimize.minimize(J, self.W, method='L-BFGS-B', jac=True,
                                         options={'maxiter': self.opts['maxIter'], 'disp': True})
        self.W = result.x  # save the optimal solution found

    def predict(self, X):
        """
        Use W to predict the classes of each example in X.

        X:              (M x N) matrix of input feature values,
                            where M = exSize, N = number of examples
        """

        W = self.W.reshape(self.numClasses, self.exSize)
        W_X = W.dot(X)

        # TODO: Compute the predicted probabilities and the predicted classes for each example
        # Reminder: The predicted class for a single example is just the one with the highest probability

        ### YOUR CODE HERE ###
        # There is no need to compute the probabilities if we only want to predict the labels of the examples.
        # This is because within the same column of W_X, the element of the largest numerical value will surely
        # be converted to the largest probability. Therefore, we can simply find the largest elements
        # in each of the columns of W_X and return their row indices.
        predicted_classes = W_X.argmax(axis=0)

        ### Below is the implementation as instructed. It's less efficient. ###
        # probabilities = np.exp(W_X)
        # probabilities = probabilities / probabilities.sum(axis=0)
        # predicted_classes = probabilities.argmax(axis=0)
        ### YOUR CODE (ENDS) HERE ###

        return predicted_classes
