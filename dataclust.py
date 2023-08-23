"""
Author: Dhruv Dhar
email: dhruvdhar1@gmail.com
"""
import numpy as np
from scipy.stats import multivariate_normal

def kmeans(X, num_clusters, max_iterations = 100):
    """
    Kmeans implementation using an Expectation-Maximization algorithm.
    Returns labels and label centroids after EM convergence
    
    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory
            copy if the given data is not C-contiguous.
            If a sparse matrix is passed, a copy will be made if it's not in
            CSR format.

    num_clusters: The number of clusters to form as well as the number of
        centroids to generate.

    max_iterations: maximum number of iterations for which the EM loop will run

    """
    centroids = X[np.random.choice(X.shape[0], num_clusters, replace=False)]
    for _ in range(max_iterations):

        #E-step
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        #M-step
        new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(num_clusters)])

        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return labels, centroids


class GaussianMixtureModel:

    """
    Gaussian Mixture Model implementation in python. This implementations is inspired from Oran looney's implementation
    (ref. https://www.oranlooney.com/post/ml-from-scratch-part-5-gmm/)

    Parameters
    ----------
    k: The number of solf clusters to form
    
    max_iter: maximum number of iterations for which the EM loop will run
    """

    def __init__(self, k=2, max_iter=50):
        self.k = k
        self.max_iter = max_iter

    def fit_transform(self, X):
        self.num_samples, self.num_features = X.shape
        
        #initialization
        self.pi = np.full(shape=(self.num_samples, self.k), fill_value=1)
        self.weights = np.full(shape=(self.num_samples, self.k), fill_value=1)
        self.mean = X[np.random.choice(self.num_samples, self.k, replace=False)]
        self.covariance = [ [1,1] for i in range(self.k) ]

        for i in range(self.max_iter):
            self.e_step(X)
            self.m_step(X)

        return self.mean, self.covariance, self.pi

    def e_step(self, X):
        likelihood = np.zeros( (self.num_samples, self.k) )
        for m in range(self.k):
            likelihood[:,m] = multivariate_normal(mean=self.mean[m], cov=self.covariance[m]).pdf(X)
        self.weights = (likelihood * self.pi)/(likelihood * self.pi).sum(axis=1)[:, np.newaxis]
        self.pi = self.weights.mean(axis=0)
        

    def m_step(self, X):
        for i in range(self.k):
            weight = self.weights[:, [i]]
            total_weight = weight.sum()

            self.covariance[i] = np.cov(X.T, 
                aweights=(weight/total_weight).flatten(), 
                bias=True)
            self.mean[i] = (X * weight).sum(axis=0) / total_weight
            
