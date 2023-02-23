import numpy as np

class Perceptron:
    
    def __init__(self):
        self.w = []
        self.accuracy = 0
        self.history = []

    def fit(self, X, y, max_steps = 1000):
        """
        Take the feature matrix X, label vector y, and maximum steps for update (optional).
        Fit the perceptron to data, pass the weight vector to self.w, current accuracy to self.accuracy, 
        and history of accuracy to self.history. 
        """
        # determine n and p from X
        n = X.shape[0] # number of data points (number of rows)
        p = X.shape[1] # number of features (number of columns)
        
        # modify X into X_
        X_ = np.append(X, np.ones((X.shape[0], 1)), 1)
        
        # modify y into an array y_ of -1s and 1s
        y_ = 2*y - 1

        # initialize random weight vector w
        self.w = np.random.uniform(-1, 1, size = (p+1,))
        
        for s in range(max_steps):
            
            # generate random index i between 0 and n-1
            i = np.random.randint(n-1)

            # extract the ith row of X_
            x_ = X_[i]

            # the ith element of y_
            y_i = y_[i]

            # compute predicted value
            y_hat = 2 * ((self.w@x_) >= 0 ) - 1

            # perform update if y_i*y_hat < 0
            self.w = self.w + (y_i*y_hat < 0) * (y_i * x_)

            # compute accuracy
            self.accuracy = (np.sum(y == ((X_@self.w) >= 0))) / n

            # append accuracy to history
            self.history.append(self.accuracy)

            # break loop if accuracy reaches 1
            if self.accuracy == 1:
                break
        
    def predict(self, X):
        """
        Take a feature matrix X and return a vector of predicted labels from the matrix. 
        """
        # append a column of 1s to X
        X_ = np.append(X, np.ones((X.shape[0], 1)), 1)
        return 1 * (X_@self.w >= 0)
            
    def score(self, X, y):
        """
        Take a feature matrix X and a label vector y. Return the accuracy of the perceptron as
        a number between 0-1, with 1 corresponding to perfect classification. 
        """
        n = X.shape[0] # number of data points
        y_hat = self.predict(X)
        return (np.sum(y == y_hat)) / n
        