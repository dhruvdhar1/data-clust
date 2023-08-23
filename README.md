# data-clust
### v1

This GitHub repository is a go-to resource for a collection of open-source unsupervised machine learning libraries written from scratch. These libraries provide a diverse set of tools to help you uncover hidden patterns, discover valuable insights, and solve complex problems in the realm of unsupervised machine learning. This library contains the following Machien learning libraries:
- K Means clustering
- Gaussian Mixture model
- DB Scan
- Hierarchical clustering
- Principal Component Analysis (PCA)
- kernel PCA
- t-SNE
- KL Summarization
- Uniform Sampling
- 1d Gaussian Sampling
- 2d Gaussian Sampling
- Gibbs Sampling
- Latent Dirichlet allocation (LDA)

## Usage

### KMeans

The below code block demonstrates usage using the MNIST dataset

```python
from keras.datasets import mnist
import numpy as np
from dataclust import kmeans

(trainX, trainY), (testX, testY) = keras.datasets.mnist.load_data()

#reshaping images
trainX = np.reshape(trainX, (-1, 784))
testX = np.reshape(testX, (-1, 784))

# normalize
trainX = trainX.astype('float32') / 255
testX = testX.astype('float32') / 255

labels, centroids = kmeans(trainX, 10)
```