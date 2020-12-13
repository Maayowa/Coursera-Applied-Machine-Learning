class KNN():
    """
    This is an attempt at building a KNN classifier from scratch.
    
    It appears the code does not respond to change in number of neighbors
    
    Would hopefully work on it with time"""
    
    import numpy as np

    
    def __init__(self):
        pass
        
    
    def fit(self, X,y, n_neighbors = 3):
        """
        ----Provide training data----
        X: contains all the features of the train data
        y: contains labels of train data
                    ---
        """
        self.X_train = X
        self.y_train = y
        self.k = n_neighbors
        
        try:
            assert self.k < len(self.y_train) + 1
            
            return
        except AssertionError:
            print("Assertion Error: n_neighbors to be used for prediction must be 1 greater than feature labels")
            
        
        
    def predict(self, X):
        """
        ---Provide featureset to be predicted---
        """
        import numpy as np
        
        # Creating an empty array containing arbitrary values as long as the test size
        y_pred = np.zeros(len(X), dtype = self.y_train.dtype)
        
        # Unique list of labels to predict
        labels = np.unique(self.y_train).tolist()

        # Looping through individual rows in the test set data
        for idx, feature in enumerate(X):
            # Using ditionaries to store predictions would not work because you can't have multiple key values 
            # with the same name in a dictionary
            
            train_pred = {label:0 for label in labels}
            distance = np.zeros(len(self.X_train))
            
            # Looking fo the nearest neigihbors in the training set
            for idxs,train in enumerate(self.X_train):
                 # Calculate euclidean distance and add it to distances
                distance[idxs] = np.linalg.norm(np.array(train) - np.array(feature))
                    
            least_distances = sorted(distance)[:self.k]
            label_idx = [np.where((distance == value))[0][0] for value in least_distances]
            likely_pred = [self.y_train.tolist()[lab_idx] for lab_idx in label_idx]
            
            # Evaluating prediction
            
            for label in likely_pred:
                train_pred[label] += 1
                
            y_pred[idx] = max(train_pred)
            #y_conf[idx] = [train_pred[i] / self.k for i in train_pred]
            
        
        return y_pred
    
    
    def predict_proba(self, X):
        
        import numpy as np
        
        y_conf = np.zeros(len(X), dtype = self.y_train.dtype)

        labels = np.unique(self.y_train).tolist()

        
        for idx, feature in enumerate(X):
            # Using ditionaries to stor predictions would not work because you can't have multiple key values 
            # with the same name in a dictionary
            
            train_pred = {label:0 for label in labels}
            distance = np.zeros(len(self.X_train))
            for idxs,train in enumerate(self.X_train):
                 # Calculate euclidean distance and add it to distances
                    distance[idxs] = np.linalg.norm(np.array(train) - np.array(feature))
                    
            least_distances = sorted(distance)[:self.k]
            label_idx = [np.where((distance == value))[0][0] for value in least_distances]
            
            likely_pred = [self.y_train.tolist()[lab_idx] for lab_idx in label_idx]
            
            # Evaluating prediction
            
            for label in likely_pred:
                train_pred[label] += 1
                
            y_conf[idx] = [train_pred[i] / self.k for i in train_pred]
            
        return y_conf
    
    def score(self, y_pred, y_test):
        """
        Conpare predicted value against labeled data
        """
        try:
            
            assert len(y_pred) == len(y_test)
            correct = 0
            for i in range(len(y_pred)):
                if y_pred[i] == y_test[i]:
                    correct += 1
                    
            return correct/len(y_pred)
        except:
            print('AssertionError: Length of test data {} must be the same length as test data {}'.format(y_pred.shape, y_test.shape))
        
        