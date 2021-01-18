import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier

class RandomForest(object):
    def __init__(self, n_estimators=50, max_depth=None, max_features=0.7):
        # helper function. You don't have to modify it
        # Initialization done here
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.bootstraps_row_indices = []
        self.feature_indices = []
        self.out_of_bag = []
        self.decision_trees = [sklearn.tree.DecisionTreeClassifier(max_depth=max_depth, criterion='entropy') for i in range(n_estimators)]
        
    def _bootstrapping(self, num_training, num_features, random_seed = None):
        """
        TODO: 
        - Randomly select indices of size num_training with replacement corresponding to row locations of 
          selected samples in the original dataset.
        - Randomly select indices without replacement corresponding the column locations of selected features in the original feature
           list (num_features denotes the total number of features in the training set, max_features denotes the percentage 
           of features that are used to fit each decision tree).
        
        Reference: https://en.wikipedia.org/wiki/Bootstrapping_(statistics)
        
        Args: 
        - num_training: an integer N representing the total number of training instances.
        - num_features: an integer D representing the total number of features.
            
        Returns:
        - row_idx: (N,) numpy array of row indices corresponding to the row locations of the selected samples in the original dataset.
        - col_idx: 1-D array of column indices corresponding to the column locations of selected features in the original feature list. 
                    
        Hint: Consider using np.random.choice.
        """
        
        np.random.seed(random_seed)
        row_idx = np.random.choice(num_training, num_training)
        real_features = int(num_features * self.max_features)
        col_idx = np.random.choice(num_features, real_features, replace = False)
        return row_idx, col_idx
            
    def bootstrapping(self, num_training, num_features):
        """
        Args: 
        - num_training: an integer N representing the total number of training instances.
        - num_features: an integer D representing the total number of features.
        
        Returns:
        - None
        """
        # helper function. You don't have to modify it
        # Initializing the bootstap datasets for each tree
        for i in range(self.n_estimators):
            total = set(list(range(num_training)))
            row_idx, col_idx = self._bootstrapping(num_training, num_features)
            total = total - set(row_idx)
            self.bootstraps_row_indices.append(row_idx)
            self.feature_indices.append(col_idx)
            self.out_of_bag.append(total)

    def fit(self, X, y):
        """
        TODO:
        Train decision trees using the bootstrapped datasets.
        Note that you need to use the row indices and column indices.
        
        Args:
        -X: NxD numpy array, where N is number 
           of instances and D is the dimensionality of each 
           instance
        -y: Nx1 numpy array, the predicted labels
        
        Returns:
        - None
        """
        N, D = X.shape
        self.bootstrapping(N, D)
        for i in range(self.n_estimators):
            b_r_indices = self.bootstraps_row_indices[i]
            f_indices = self.feature_indices[i]
            
            train_x = X[b_r_indices][:,f_indices]
            train_y = y[b_r_indices]
            
            d_tree = self.decision_trees[i]
            
            print('fitting decision tree # ', i)
            d_tree.fit(train_x, train_y)
    
    def OOB_score(self, X, y):
        # helper function. You don't have to modify it
        # This function computes the accuracy of the random forest model predicting y given x.
        accuracy = []
        for i in range(len(X)):
            predictions = []
            for t in range(self.n_estimators):
                if i in self.out_of_bag[t]:
                    predictions.append(self.decision_trees[t].predict(np.reshape(X[i][self.feature_indices[t]], (1,-1)))[0])
            if len(predictions) > 0:
                accuracy.append(np.sum(predictions == y[i]) / float(len(predictions)))
        return np.mean(accuracy)