import numpy as np
from collections import Counter

############################################################################
# DO NOT MODIFY CODES ABOVE 
############################################################################

class KNN:
    def __init__(self, k, distance_function):
        """
        :param k: int
        :param distance_function
        """
        self.k = k
        self.distance_function = distance_function
        self.features = []
        self.labels = []

    # save features and lable to self
    def train(self, features, labels):
        """
        In this function, features is simply training data which is a 2D list with float values.
        For example, if the data looks like the following: Student 1 with features age 25, grade 3.8 and labeled as 0,
        Student 2 with features age 22, grade 3.0 and labeled as 1, then the feature data would be
        [ [25.0, 3.8], [22.0,3.0] ] and the corresponding label would be [0,1]

        For KNN, the training process is just loading of training data. Thus, all you need to do in this function
        is create some local variable in KNN class to store this data so you can use the data in later process.
        :param features: List[List[float]]
        :param labels: List[int]
        """
        self.features = features
        self.labels = labels

    # find KNN of one point
    def get_k_neighbors(self, point):
        """
        This function takes one single data point and finds k-nearest neighbours in the training set.
        You already have your k value, distance function and you just stored all training data in KNN class with the
        train function. This function needs to return a list of labels of all k neighbours.
        :param point: List[float]
        :return:  List[int]
        """
        dist = []
        knn = []
        for Li in self.features: # get distance for all feature vectors
            dist.append(self.distance_function(Li, point))
        
        # sort --> so we can find k nearest
        if self.k > len(dist):
            self.k = len(dist)
        distPair = []
        for i in range(len(dist)):
            distPair.append((self.labels[i], dist[i]))
        distPair = np.array(distPair)
        distPair = distPair[np.argsort(distPair[:,1])] #get the indicies of sorted
        for i in range(self.k):
            knn.append(distPair[i][0])
        return knn
		
	# predict labels of a list of points
    def predict(self, features):
        """
        This function takes 2D list of test data points, similar to those from train function. Here, you need to process
        every test data point, reuse the get_k_neighbours function to find the nearest k neighbours for each test
        data point, find the majority of labels for these neighbours as the predicted label for that testing data point (you can assume that k is always a odd number).
        Thus, you will get N predicted label for N test data point.
        This function needs to return a list of predicted labels for all test data points.
        :param features: List[List[float]]
        :return: List[int]
        """
        result = []
        for xi in features:
            neighbors = self.get_k_neighbors(xi)
            result.append(max(set(neighbors), key=neighbors.count))
        return result	


if __name__ == '__main__':
    print(np.__version__)
