import numpy as np


class Regression(object):

    def __init__(self):
        pass

    def rmse(self, pred, label):  # [5pts]
        '''
        This is the root mean square error.
        Args:
            pred: numpy array of length N * 1, the prediction of labels
            label: numpy array of length N * 1, the ground truth of labels
        Return:
            a float value
        '''
        return np.sqrt(((pred - label) ** 2).mean())

    def construct_polynomial_feats(self, x, degree):  # [5pts]
        """
        Args:
            x: numpy array of length N, the 1-D observations
            degree: the max polynomial degree
        Return:
            feat: numpy array of shape Nx(degree+1), remember to include
            the bias term. feat is in the format of:
            [[1.0, x1, x1^2, x1^3, ....,],
             [1.0, x2, x2^2, x2^3, ....,],
             ......
            ]
        """
        return np.power(np.repeat(x.reshape(-1, 1) ,degree + 1, axis = 1), np.arange(degree + 1))

    def predict(self, xtest, weight):  # [5pts]
        """
        Args:
            xtest: NxD numpy array, where N is number
                   of instances and D is the dimensionality of each
                   instance
            weight: Dx1 numpy array, the weights of linear regression model
        Return:
            prediction: Nx1 numpy array, the predicted labels
        """
        return xtest @ weight

    # =================
    # LINEAR REGRESSION
    # Hints: in the fit function, use close form solution of the linear regression to get weights.
    # For inverse, you can use numpy linear algebra function
    # For the predict, you need to use linear combination of data points and their weights (y = theta0*1+theta1*X1+...)

    def linear_fit_closed(self, xtrain, ytrain):  # [5pts]
        """
        Args:
            xtrain: N x D numpy array, where N is number of instances and D is the dimensionality of each instance
            ytrain: N x 1 numpy array, the true labels
        Return:
            weight: Dx1 numpy array, the weights of linear regression model
        """
        return np.linalg.pinv(xtrain.T @ xtrain) @ xtrain.T @ ytrain

    def linear_fit_GD(self, xtrain, ytrain, epochs=5, learning_rate=0.001):  # [5pts]
        """
        Args:
            xtrain: NxD numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: Nx1 numpy array, the true labels
        Return:
            weight: Dx1 numpy array, the weights of linear regression model
        """
        weight = np.zeros((xtrain.shape[1], 1))
        for i in range(epochs):
            weight += learning_rate * (xtrain.T @ (ytrain - xtrain @ weight)) / xtrain.shape[0]
        return weight

    def linear_fit_SGD(self, xtrain, ytrain, epochs=100, learning_rate=0.001):  # [5pts]
        """
        Args:
            xtrain: NxD numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: Nx1 numpy array, the true labels
        Return:
            weight: Dx1 numpy array, the weights of linear regression model
        """
        weight = np.zeros((xtrain.shape[1], 1))
        for i in range(epochs):
            weight += learning_rate * (xtrain.T @ (ytrain - xtrain @ weight)) / xtrain.shape[0]
        return weight

    # =================
    # RIDGE REGRESSION

    def ridge_fit_closed(self, xtrain, ytrain, c_lambda):  # [5pts]
        """
        Args:
            xtrain: N x D numpy array, where N is number of instances and D is the dimensionality of each instance
            ytrain: N x 1 numpy array, the true labels
            c_lambda: floating number
        Return:
            weight: Dx1 numpy array, the weights of ridge regression model
        """
        identity = np.identity(xtrain.shape[1])
        identity[: 1] = 0
        return np.linalg.pinv(xtrain.T @ xtrain + c_lambda * identity) @ xtrain.T @ ytrain

    def ridge_fit_GD(self, xtrain, ytrain, c_lambda, epochs=500, learning_rate=1e-7):  # [5pts]
        """
        Args:
            xtrain: NxD numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: Nx1 numpy array, the true labels
            c_lambda: floating number
        Return:
            weight: Dx1 numpy array, the weights of linear regression model
        """
        identity = np.identity(xtrain.shape[1])
        identity[: 1] = 0
        idt = identity * learning_rate * c_lambda
        weight = np.zeros((xtrain.shape[1], 1))
        for i in range(epochs):
            idt_w = idt @ weight
            weight += learning_rate * (xtrain.T @ (ytrain - xtrain @ weight)) / xtrain.shape[0] - idt_w
        return weight

    def ridge_fit_SGD(self, xtrain, ytrain, c_lambda, epochs=100, learning_rate=0.001):  # [5pts]
        """
        Args:
            xtrain: NxD numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: Nx1 numpy array, the true labels
        Return:
            weight: Dx1 numpy array, the weights of linear regression model
        """
        identity = np.identity(xtrain.shape[1])
        identity[: 1] = 0
        idt = identity * learning_rate * c_lambda * 2
        weight = np.zeros((xtrain.shape[1], 1))
        for i in range(epochs):
            for j in range(xtrain.shape[0]):
                idt_w = idt @ weight
                weight += learning_rate * (((xtrain * ytrain).T)[:, j, np.newaxis] - \
                    (xtrain[:, :, np.newaxis] @ xtrain[:, np.newaxis, :])[j] @ weight) - idt_w
        return weight

    def ridge_cross_validation(self, X, y, kfold=10, c_lambda=100):  # [8 pts]
        """
        Args: 
            X : NxD numpy array, where N is the number of instances and D is the dimensionality of each instance
            y : Nx1 numpy array, true labels
            kfold: Number of folds you should take while implementing cross validation.
            c_lambda: Value of regularization constant
        Returns:
            meanErrors: Float average rmse error
        Hint: np.concatenate might be helpful.
        Look at 3.5 to see how this function is being used.
        # For cross validation, use 10-fold method and only use it for your training data (you already have the train_indices to get training data).
        # For the training data, split them in 10 folds which means that use 10 percent of training data for test and 90 percent for training.
        """
        error = []
        for i in range(kfold):
            x_train = (np.split(X, kfold)).copy()
            xtest = x_train.pop(i)
            y_train = (np.split(y, kfold)).copy()
            ytest = y_train.pop(i)
            xtrain = np.vstack(x_train)
            ytrain = np.vstack(y_train)
            weight = self.ridge_fit_closed(xtrain, ytrain, c_lambda)
            prediction = self.predict(xtest, weight)
            rmse = self.rmse(prediction, ytest)
            error.append(rmse)
        return np.mean(error)
