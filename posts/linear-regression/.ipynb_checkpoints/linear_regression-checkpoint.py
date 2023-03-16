import numpy as np

class LinearRegression:
    
    def __init__(self):
        self.w = []
        self.accuracy = 0
        self.score_history = []
        
    def pad(self, X):
        """
        Take a matrix X and append a column of 1s to the end. 
        """
        return np.append(X, np.ones((X.shape[0], 1)), 1)
    
    def gradient(self, P, q, w):
        """
        Compute the gradient for the loss function of linear regression associated with w, X, and y. 
        """
        return P@w - q

    def fit(self, X, y, method = "analytic", alpha = 0.001, max_iter = 1000):
        """
        Take a feature matrix X and a label vector y and fit the logistic regression model to data using
        gradient descent. 
        
        Optional parameters:
        - alpha: learning rate
        - max_epochs: max steps of gradient update
        """
        X_ = self.pad(X)
    
        # analytic method
        if method == "analytic":
            self.w = np.linalg.inv(X_.T@X_)@X_.T@y
        
        # gradient descent method
        if method == "gradient":
            # determine n and p from X
            n = X_.shape[0] 
            p = X_.shape[1]

            # pick a random weight vector
            self.w = .5 - np.random.rand(p)

            # initialize prev_loss
            prev_loss = np.inf

            # calculate P and q for gradient descent
            P = X_.T@X_
            q = X_.T@y
            
            # main loop
            for i in np.arange(max_iter): 
                self.w -= 2*alpha*self.gradient(P, q, self.w)

                # compute loss
                new_loss = np.linalg.norm(X_@self.w - y, 2) ** 2
            
                # compute accuracy and append to history
                self.accuracy = self.score(X, y)
                self.score_history.append(self.accuracy)

                # check if loss hasn't changed and terminate if so
                if np.isclose(new_loss, prev_loss):          
                    break
                else:
                    prev_loss = new_loss
        
        
        
    def predict(self, X):
        """
        Take a feature matrix X and return a vector of predicted values using the current model. 
        """
        X_ = self.pad(X)
        return X_@self.w
    
    def score(self, X, y):
        """
        Take a feature matrix X and a label vector y and return the accuracy score (0-1) of the current
        model. 
        """
        y_hat = self.predict(X)
        y_bar = y.mean()
        return 1 - ((y_hat - y) ** 2).sum() / ((y_bar - y) ** 2).sum()
        
    
    
    
        