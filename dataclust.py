"""
Author: Dhruv Dhar
email: dhruvdhar1@gmail.com
"""
import random
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

def check_core_point(eps,minPts, df, index):
    x, y = df.iloc[index]['x']  ,  df.iloc[index]['y']
    temp =  df[((np.abs(x - df['x']) <= eps) & (np.abs(y - df['y']) <= eps)) & (df.index != index)]
    
    if len(temp) >= minPts:
        return (temp.index , 1)
    
    elif (len(temp) < minPts) and len(temp) > 0:
        return (temp.index , 2)
    
    elif len(temp) == 0:
        return (temp.index , 3)

def dbscan(eps, minPts, df):
    #initiating cluster number
    cluster_num = 0

    q = set()
    unvisited = list(df.index)
    clusters = []
    
    while (len(unvisited) > 0): #run until all points have been visited

        #identifier for first point of a cluster
        first_point = True
        
        #choose a random unvisited point
        q.add(random.choice(unvisited))
        
        while len(q) > 0:
            pop = q.pop()
            unvisited.remove(pop)
            neighbor_ind, point_type = check_core_point(eps, minPts, df, pop)
            
            #dealing with an edge case
            if point_type == 2 and first_point:
                
                clusters.append((pop, -1))
                for ind in neighbor_ind:
                    clusters.append((ind, -1))
                unvisited = [element for element in unvisited if element not in neighbor_ind]
                continue
            first_point = False
            
            #CORE POINT
            if point_type == 1:
                clusters.append((pop,cluster_num))
                neighbor_ind = set(neighbor_ind) & set(unvisited)
                q.update(neighbor_ind)

            #BORDER POINT
            elif point_type == 2:
                clusters.append((pop,cluster_num))
            
            #OUTLIER
            elif point_type == 3:
                clusters.append((pop, -1))
                
        if not first_point:
            cluster_num += 1
        
    return clusters

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
            

def euclidean_distances(X):
    euclidean_distance_mat = np.zeros((len(X), len(X)))

    for i in range(len(X)):
        for j in range(len(X)):
            euclidean_distance = np.linalg.norm(X[i] - X[j])
            euclidean_distance_mat[i][j] = euclidean_distance
    return euclidean_distance_mat

def expand_cluster(point1, point2, m, clusters):
    expanded_list = set()
    cluster_value_set = set()
    for key in clusters:
        clust_values = clusters[key]
        cluster_value_set = cluster_value_set.union(clust_values)

    if point1 not in cluster_value_set and point2 not in cluster_value_set:
        expanded_list.add(point1)
        expanded_list.add(point2)
    else:
        for key in clusters:
            clust_values = clusters[key]
            if point1 in clust_values:
                expanded_list = expanded_list.union(clust_values)
            else:
                expanded_list.add(point1)
            if point2 in clust_values:
                expanded_list = expanded_list.union(clust_values)
            else:
                expanded_list.add(point2)
    clusters[m] = expanded_list

def hierarchical_clust(X, k):
    clusters = dict()
    #Step-1: calculate euclidean distances, replace all zero with inf for ease
    euclidean_distances_mat = euclidean_distances(X)
    euclidean_distances_mat = np.tril(euclidean_distances_mat)
    euclidean_distances_mat[euclidean_distances_mat == 0] = np.inf

    m = euclidean_distances_mat.shape[0]
    num_points = euclidean_distances_mat.shape[0]

    #Step-2: run a while loop until all no non-inf distance is present in the dist_mat
    while m > 0: 
        min_ind = np.unravel_index(euclidean_distances_mat.argmin(), euclidean_distances_mat.shape)
        # min_ind will be in the form (x,y) -> we process along x
        
        x = min_ind[0]
        y = min_ind[1]

        expand_cluster(x, y, m, clusters)

        m -= 1

        for i in range(0, num_points):
            if i != x and i != y:
                dist_x_i = euclidean_distances_mat[x][i] if euclidean_distances_mat[x][i] != np.inf else euclidean_distances_mat[i][x]
                dist_y_i = euclidean_distances_mat[y][i] if euclidean_distances_mat[y][i] != np.inf else euclidean_distances_mat[i][y]
                min_val_between_points = max(dist_x_i, dist_y_i)
                min_val_between_points = dist_x_i
                euclidean_distances_mat[x][i] = min_val_between_points

        #Step-3: Set y rows and col in distance_mat to infifity to avoid recomputation
        euclidean_distances_mat[y] = np.inf
        euclidean_distances_mat[:,y] = np.inf
    return clusters