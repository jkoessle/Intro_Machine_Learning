from collections import Counter
import operator
import numpy as np
import pandas as pd
from scipy import linalg
import matplotlib.pyplot as plt
from scipy.stats import mode
import scipy.spatial as spa

IMAGE_PATH = ""
LABEL_PATH = ""


def train_test_split(data,labels):
    # split data and labels in half
    X_train, X_test = np.split(data, 2)
    y_train, y_test = np.split(labels, 2)
    
    return X_train,X_test,y_train,y_test


def calculate_matrices(X,y):
    # get parameters
    N, D = X.shape
    K = np.max(y)
    
    # calculate S_W matrix
    SW = np.zeros((D,D))
    for k in range(1, K+1):
        class_SW = np.cov(X[y==k].T)*(len(X[y==k])-1)
        SW += class_SW
        
    # S_W without loops, unfortunately unprecise
    # SW = (X-X.mean(axis=0)).T.dot((X-X.mean(axis=0)))
    
    # calculate mean vectors for each feature
    mean_vectors = []
    for k in range(1, K+1):   
        mean_vectors.append(np.mean(X[y_train==k], axis=0)) 
           
    # calculate overall mean
    mean = np.mean(X, axis=0)
    
    # calculate S_B matrix
    SB = np.zeros((D, D))
    for i, mean_vector in enumerate(mean_vectors):
        n = len(X[y==i+1, :])
        mean_vector = mean_vector.reshape(D, 1)
        mean = mean.reshape(D, 1)
        SB += n * (mean_vector - mean).dot((mean_vector - mean).T)
    
    return SW,SB


def calculate_eig(SW,SB):
    # calculate eigenvalues and vectors
    w, v = np.linalg.eig(np.linalg.inv(SW).dot(SB))
    
    # cast values to real format to discard imaginary part
    w = w.real
    v = v.real
    
    return w,v


def draw_eigenvectors(X,y, vectors):
    K = np.max(y)
    
    # calculate two-dimensional projections
    Z = np.matmul(X - np.mean(X, axis = 0), vectors[:,[0, 1]])

    # plot two-dimensional projections
    plt.figure(figsize = (10, 10))
    point_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6"])
    for c in range(K):
        plt.plot(Z[y == c + 1, 0], Z[y == c + 1, 1], marker = "o", markersize = 4, linestyle = "none", color = point_colors[c])
    plt.legend(["t-shirt/top", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"],
            loc = "upper left", markerscale = 2)
    plt.xlabel("Component #1")
    plt.ylabel("Component #2")
    ax = plt.gca()
    ax.set_ylim([-6.0, 6.0])
    ax.set_xlim([-6.0, 6.0])
    plt.show()


def draw_confusion_matrix(y, y_hat):
    N = len(y)
    confusion_matrix= pd.crosstab(np.reshape(y_hat, N), y,
                               rownames = ["y_predicted"], colnames = ["y_train"])
    print("Confusion Matrix:")
    print(confusion_matrix)
    print()


def kNN(X_1,X_2,y,k):
    # calculate distances between each entry of X_1 and X_2
    distances = spa.distance_matrix(X_1,X_2)
    
    # get indices of corresponding k-th entries
    index = np.argpartition(distances, k, axis=0)[:k]
    
    # find neighbors
    neighbors = np.take(y, index)
    
    # predict class label
    prediction = mode(neighbors,axis=0)[0]
    
    return prediction


if __name__ == "__main__":
    # read data into memory
    image_data = np.genfromtxt(IMAGE_PATH, delimiter = ",")
    image_labels = np.genfromtxt(LABEL_PATH, delimiter = ",").astype(int)
    
    # split train/test data
    X_train, X_test, y_train, y_test = train_test_split(image_data,image_labels)
    
    # calculate S_W and S_B matrices
    SW,SB = calculate_matrices(X_train,y_train)
    print("First five entries for SW matrix:")
    print(SW[0:5, 0:5])
    print()
    print("First five entries for SB matrix:")
    print(SB[0:5, 0:5])
    print()
    
    # # calculate eigenvalues and vectors
    values, vectors = calculate_eig(SW,SB)
    print("Nine largest eigenvalues:")
    print(values[0:9])
    # print("Corresponding eigenvectors:")
    # print(vectors[0:9])
    print()
    
    
    # visualizations of eigenvectors
    draw_eigenvectors(X_train,y_train,vectors)
    draw_eigenvectors(X_test,y_test,vectors)
    
    # compute nine-dimensional projections for train and test data
    Z_train = np.matmul(X_train - np.mean(X_train, axis = 0), vectors[:,0:9])
    Z_test = np.matmul(X_test - np.mean(X_test, axis = 0), vectors[:,0:9])
    
    # compute nearest neighbors for train and test data
    y_hat_train = kNN(Z_train,Z_train,y_train,k=11)
    y_hat_test = kNN(Z_train,Z_test,y_train,k=11)
    
    # draw confusion matrices
    print("Confusion matrix for training data:")
    draw_confusion_matrix(y_train,y_hat_train)
    print("Confusion matrix for test data:")
    draw_confusion_matrix(y_test,y_hat_test)