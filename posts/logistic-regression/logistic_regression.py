import numpy as np

class LogisticRegression:
    
    def __init__(self):
        self.w = []
        self.accuracy = 0
        self.loss_score = np.inf
        self.score_history = []
        self.loss_history = []
        
    def pad(self, X):
        """
        Take a matrix X and append a column of 1s to the end. 
        """
        return np.append(X, np.ones((X.shape[0], 1)), 1)

    def sigmoid(self, z):
        """
        Sigmoid function. 
        Source: CS451 lecture note
        """
        return 1 / (1 + np.exp(-z))

    def logistic_loss(self, y_hat, y): 
        """
        Take an array of predicted value and an array of labels and compute the logistic loss. 
        Source: CS451 lecture note
        """
        return -y*np.log(self.sigmoid(y_hat)) - (1-y)*np.log(1-self.sigmoid(y_hat))

    def empirical_risk(self, X, y, loss, w):
        """
        Take a feature matrix X, label vector y, a loss function, and a weight vector and return the 
        empirical risk. 
        Source: CS451 lecture note
        """
        y_hat = X@w
        return loss(y_hat, y).mean()
    
    def gradient(self, w, X_, y):
        """
        Compute the gradient for the logistic loss function associated with w, X_, and y. 
        """
        # transform arrays of shape (n,) to arrays of shape(n, 1)
        w = w[:,np.newaxis]
        y = y[:,np.newaxis]
        
        # gradient of logistic loss
        return np.mean((self.sigmoid(X_@w) - y) * X_, axis = 0)

    def fit(self, X, y, alpha = 0.1, max_epochs = 1000):
        """
        Take a feature matrix X and a label vector y and fit the logistic regression model to data using
        gradient descent. 
        
        Optional parameters:
        - alpha: learning rate
        - max_epochs: max steps of gradient update
        """
        # reset loss and accuracy
        self.accuracy = 0
        self.loss_score = np.inf
        self.score_history = []
        self.loss_history = []
        
        # determine n and p from X
        n = X.shape[0] # number of data points (number of rows)
        p = X.shape[1] # number of features (number of columns)

        # add a constant feature to the feature matrix
        X_ = self.pad(X)
        
        # initialize prev_loss
        prev_loss = np.inf

        # pick a random weight vector
        self.w = .5 - np.random.rand(p+1)

        # main loop
        for i in np.arange(max_epochs): 
            self.w -= alpha*self.gradient(self.w, X_, y)
            
            # compute loss and append to history
            new_loss = self.empirical_risk(X_, y, self.logistic_loss, self.w)
            self.loss_score = new_loss
            self.loss_history.append(new_loss)
            
            # compute accuracy and append to history
            self.accuracy = (np.sum(y == ((X_@self.w) >= 0))) / n
            self.score_history.append(self.accuracy)
            
            # check if loss hasn't changed and terminate if so
            if np.isclose(new_loss, prev_loss):          
                break
            else:
                prev_loss = new_loss
                
    def predict(self, X):
        """
        Take a feature matrix X and return a vector of labels predicted using the current model. 
        """
        X_ = self.pad(X)
        return 1 * (X_@self.w >= 0)
    
    def score(self, X, y):
        """
        Take a feature matrix X and a label vector y and return the accuracy score (0-1) of the current
        model. 
        """
        n = X.shape[0] # number of data points
        y_hat = self.predict(X)
        return (np.sum(y == y_hat)) / n
        
    def loss(self, X, y):
        """
        Take a feature matrix X and a label vector y and return the overall loss (empirical risk) of the
        current weights on X and y. 
        """
        X_ = self.pad(X)
        return self.empirical_risk(X_, y, self.logistic_loss, self.w)
    
    def fit_stochastic(self, X, y, alpha = 0.1, max_epochs = 1000, batch_size = 10):
        """
        Take a feature matrix X and a label vector y and fit the logistic regression model to data using
        stochastic gradient descent. 
        
        Optional parameters:
        - alpha: learning rate
        - max_epochs: max steps of gradient update
        - batch_size: size of each batch created for stochastic gradient descent
        """
        
        # reset loss and accuracy
        self.accuracy = 0
        self.loss_score = np.inf
        self.score_history = []
        self.loss_history = []
        
        # determine number of data points and features
        n = X.shape[0]
        p = X.shape[1]
        
        # transform X
        X_ = self.pad(X)
        
        # pick a random weight vector
        self.w = .5 - np.random.rand(p+1)
        
        # intialize prev_loss
        prev_loss = np.inf
 
        # main loop
        for j in np.arange(max_epochs):

            order = np.arange(n)
            np.random.shuffle(order)
 
            for batch in np.array_split(order, n // batch_size + 1):
        
                # get and transform feature matrix for the batch
                x_batch = X[batch,:]
                x_batch = self.pad(x_batch)
                # labels for the batch
                y_batch = y[batch] 

                # gradient step
                self.w -= alpha*self.gradient(self.w, x_batch, y_batch)

            # after each epoch, compute loss and append to history
            new_loss = self.empirical_risk(X_, y, self.logistic_loss, self.w)
            self.loss_score = new_loss
            self.loss_history.append(self.loss_score)
                
            # compute accuracy and append to history
            self.accuracy = (np.sum(y == ((X_@self.w) >= 0))) / n
            self.score_history.append(self.accuracy)

            # check if loss hasn't changed and terminate if so
            if np.isclose(new_loss, prev_loss):          
                break
            else:
                prev_loss = new_loss
                    
        
                
    
    
    
        