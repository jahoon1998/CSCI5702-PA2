# Oct 25 2019 CSCI 5702 Big Data Mining  
# Assignment 2: K-Means & Latent Factors

## Part 1: K-Means


This problem will help you understand the nitty-gritty details of implementing clustering algorithms on Spark. In addition, this problem will also help you understand the impact of using various distance metrics and initialization strategies in practice. You must not use the Spark MLlib clustering library (or similar libraries) for this problem.

Let us say we have a set X of n data points in the d-dimensional space Rd. Given the number of clusters k and the set of k centroids C, we now proceed to define various distance metrics and the corresponding cost functions that they minimize.

### Euclidean distance 
Given two points A and B in d dimensional space such that A = [a<sub>1</sub>,a<sub>2</sub> ···a<sub>d</sub>] and B = [b<sub>1</sub>,b<sub>2</sub> ···b<sub>d</sub>], the Euclidean distance between A and B is defined as: ![equation](https://latex.codecogs.com/gif.latex?\left&space;\|&space;a&space;-&space;b&space;\right&space;\|&space;=&space;\sqrt{\sum_{i=1}^{d}(a_{i}-b_{i})^{^{2}}})

The corresponding cost function φ that is minimized when we assign points to clusters using the Euclidean distance metric is given by: 
<img src="https://i.postimg.cc/2yR2qFHF/Euclidea-Distance.png" width="20%"></img>

### Manhattan distance
Given two random points A and B in d dimensional space such that A = [a<sub>1</sub>,a<sub>2</sub> ···a<sub>d</sub>] and B = [b<sub>1</sub>,b<sub>2</sub> ···b<sub>d</sub>], the Manhattan distance between A and B is defined as:
<img src="https://i.postimg.cc/nVB3ygzd/manhattan-distance.png" width="20%"></img>

The corresponding cost function ψ that is minimized when we assign points to clusters using the Manhattan distance metric is given by:
<img src="https://i.postimg.cc/8PGZN4fT/MD.png" width="40%"></img>

### Iterative k-Means Algorithm: 
We learned the basic k-Means algorithm in class which is as follows: k centroids are initialized, each point is assigned to the nearest centroid and the centroids are recomputed based on the assignments of points to clusters. In practice, the above steps are run for several iterations. We present the resulting iterative version of k-Means in Algorithm 1.
<img src="https://i.postimg.cc/sDJwXW57/iterative-kmeans-algorithm.png" width="60%"></img>

### Iterative k-Means clustering on Spark: 
Implement iterative k-means using Spark. Note that we have provided that centroids for you (see below), so you do not need to select the initial centroids (skip Line 2 of Algorithm 1).

### P1 has 3 files:
1. data.txt contains the dataset which has 4601 rows and 58 columns. Each row is a document represented as a 58-dimensional vector of features. Each component in the vector represents the importance of a word in the document.
2. c1.txt contains k initial cluster centroids. These centroids were chosen by selecting k = 10 random points from the input data.
3. c2.txt contains initial cluster centroids which are as far apart as possible. (You can do this by choosing 1st centroid c1 randomly, and then finding the point c2 that is farthest from c1, then selecting c3 which is farthest from c1 and c2, and so on).

Set the number of iterations to 20 and the number of clusters k to 10 for all the experiments carried out in this question. Your driver program should ensure that the correct amount of iterations are run.

### a. Exploring initialization strategies with Euclidean distance
Using the Euclidean distance (refer to Equation 1) as the distance measure, compute the cost function φ(i) (refer to Equation 2) for every iteration i. This means that, for your first iteration, you will be computing the cost function using the initial centroids located in one of the two text files. Run the k-means on data.txt using c1.txt and c2.txt. Generate a graph (line plot) where you plot the cost function φ(i) as a function of the number of iterations i=1..20 for c1.txt and also for c2.txt.
#### Hint: 
Note that you do not need to write a separate Spark job to compute φ(i). You should be able to calculate costs while partitioning points into clusters.

### b. Exploring initialization strategies with Manhattan distance
Using the Manhattan distance metric (refer to Equation 3) as the distance measure, compute the cost function ψ(i) (refer to Equation 4) for every iteration i. This means that, for your first iteration, you’ll be computing the cost function using the initial centroids located in one of the two text files. Run the k-means on data.txt using c1.txt and c2.txt. Generate a graph where you plot the cost function ψ(i) as a function of the number of iterations i=1..20 for c1.txt and also for c2.txt.

### Code Structure:
Your code must have a kmeans function with the following signature (you can define as many functions as you like, but the kmeans function is the driver function meaning that it should be the only function that is directly called by the user):

```python
kmeans(data, centroids, iterations, euclidean_distance)
```
* `data` is an RDD. Each value of data must be a numpy array of type float, containing all the values for one line on the input file. For example, if the input line is 0 3.245 3.2 your array must contain [0, 3.245. 3.2]
* `centroids` is an RDD. Each value of centroids must be a numpy array of type float, containing all the values for one line on the centroid file. For example, if the input line is 0 3.245 3.2 your array must contain [0, 3.245. 3.2]
* `euclidean_distance` is a Boolean variable. If it is True (False), your k-means function must use the Euclidean (Manhattan) distance as the distance measure.

You must declare four variables in the very beginning of your code:
```python
Iterations = 20 # determines the number of iterations
dataset_path = ‘’ # this is the absolute path to the dataset file
centroids_path = ‘’ # this contains the absolute path the centroid file that being use (either c1.txt or c2.txt)
euclidean_distance = True # or False
```

These variables must be immediately followed by calling the kmeans function, which outputs the cost for each of the 20 iterations in the following format (note that the values in the below example are synthetic): 
`kmeans(data, centroids, max_iterations, euclidean)`

## Part 2: Latent Factors for Recommendations
The goal of this problem is to implement the Stochastic Gradient Descent algorithm to build a Latent Factor Recommendation system. We can use it to recommend movies to users. We encourage you to read the slides of the lecture “Recommender Systems 2” again before attempting the problem.

Suppose we are given a matrix R of recommendations. The element Riu of this matrix corresponds to the rating given by user u to item i. The size of R is m × n, where m is the number of movies and n the number of users.

Most of the elements of the matrix are unknown because each user can only rate a few movies.

Our goal is to find two matrices P and Q, such that R = QP<sup>T</sup> . The dimensions of Q are m × k, and the dimensions of P are n × k. k is a parameter of the algorithm. We define the error as
<img src="https://i.postimg.cc/nr4Tqjg8/latent-factors-for-recommendations.png" width="50%"></img>

### Implementation
Implement the algorithm. Read each entry of the matrix R from disk and update ε<sub>iu</sub>, q<sub>i</sub> and p<sub>u</sub> for each entry.

To emphasize, you are not allowed to store the matrix R in memory. You have to read each element Riu one at a time from disk and apply your update equations (to each element). If you keep more than one element in memory at a time, you will receive no point for Part b. For example, File.ReadAllText() and File.ReadAllLines() in C# are examples of functions that read in the entire file into memory. On the other hand, a StreamReader object in C# reads one line at a time. Each iteration of the algorithm will read the whole file.

Choose k = 20, λ = 0.1 and number of iterations = 40. Find a good value for the learning rate η. Start with η = 0.1. The error E on the training set ratings.train.txt discussed below should be less than 65000 after 40 iterations.

Based on the values of η, you may encounter the following cases:
* If η is too big, the error function can converge to a high value or may not monotonically decrease. It can even diverge and make the components of vectors p and q equal to infinity.
* If η is too small, the error function will not have time to significantly decrease and reach convergence. So, it can monotonically decrease but not converge, i.e., it could have a high value after 40 iterations because it has not converged yet.

### Dataset
`ratings.csv`: this is the matrix R. Each entry is made of a movie id, user id, and a rating that is an integer between 1 and 5.

### Code Structure
You need to define the following five parameters in the beginning of your code:
* `iterations = 40`
* `k = 20`
* `regularization_factor = 0.1 #𝜆`
* `learning_rate = 0.1 # 𝜂`
* `data_path = ''`

Your code must have a driver function with the below signature, which is the only function that is directly called by the TAs for grading. However, you can use as many functions as you like.
`latent_factor_recommnder(data_path, regularization_factor, learning_rate, iterations, k)`

### Hints
* P and Q: we would like q<sub>i</sub> and p<sub>u</sub> for all users u and items i. A good way to achieve that is to initialize all elements of P and Q to random values in [0,√5/k].
* Updating the equations: in each update, we update q<sub>i</sub> using p<sub>u</sub> and p<sub>u</sub> using q<sub>i</sub>. Compute the new values for q<sub>i</sub> and p<sub>u</sub> using the old values, and then update the vectors q<sub>i</sub> and p<sub>u</sub>.
* You should compute E at the end of a full iteration of training. Computing E in pieces during the iteration is incorrect since P and Q are still being updated.
