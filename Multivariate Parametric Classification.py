import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import pandas as pd

def bivariateGauss(mu, cov, N):
    # compute Cholesky decomposition
    cho = np.linalg.cholesky(cov)
    
    # generate random samples (2-dimensional)
    samples = np.random.normal(loc=0, scale=1, size=2*N).reshape(2, N)
    
    result = mu.T + np.dot(cho, samples).T
        
    return result

def bivariateGaussNumpy(mu, cov, N):
    return np.random.multivariate_normal(mean=mu.T, cov=cov, size=N)

def bivariateGaussSciPy(mu, cov, N):
    return multivariate_normal.rvs(mean=mu.T,cov=cov,size=N)   

def generateData():
    # Set given parameters   
    class_means = np.array([([0.0,4.5]),([-4.5,-1.0]),([4.5,-1.0]),([0.0,-4.0])])
    class_covs = np.array([([3.2,0.0],[0.0,1.2]),([1.2,0.8],[0.8,1.2]),([1.2,-0.8],[-0.8,1.2]),([1.2,0.0],[0.0,3.2])])   
    class_sizes = np.array([105,145,135,115]) 
        
    # generate random samples, three different Gauss implementations available. Default is based on numpy function
    points_dict = {}
    for i in range(0,4):
        points_dict["points%s" %i] = bivariateGaussNumpy(class_means[i],class_covs[i],class_sizes[i])   
    x = np.concatenate((points_dict["points0"], points_dict["points1"], points_dict["points2"], points_dict["points3"]))

    # generate corresponding labels
    y = np.concatenate((np.repeat(1, class_sizes[0]), np.repeat(2, class_sizes[1]), np.repeat(3, class_sizes[2]),np.repeat(4, class_sizes[3])))
        
    return x,y

def fit(X, y, K, D):
    # calculate sample means
    sample_means = np.zeros((K,D))
    for k in range(K):
        points_temp = X[y == (k + 1)]
        for d in range(D):
            sample_means[k,d] = np.mean(points_temp[:,d])
            
    print("Sample Means:")
    print(sample_means)
    print()

    # calculate sample covariances
    sample_covariances = np.zeros((K,D,D))

    for k in range(K):
        points_temp = X[y == (k + 1)]
        sample_covariances[k,:,:] = np.cov(points_temp.T)
            
    print("Sample Covariances:")        
    print(sample_covariances)
    print()

    # calculate prior probabilities
    class_priors = [np.mean(y == (i + 1)) for i in range(K)]
    print("Sample Priors:")
    print(class_priors)
    print()
    
    return sample_means, sample_covariances, class_priors

def predict(X, mu, cov, prior, k):
    n = X.shape[0]
    all_predictions = np.zeros((n,k))
    prediction = np.zeros(n)
    for i in range(k):
        # compute determinant of cov matrix
        cov_det = np.log(np.linalg.det(cov[i]))
        # compute inverse of cov matrix
        cov_inv = np.linalg.inv(cov[i])
        # get log prior
        prior_log = np.log(prior[i])
        # compute pdf for every entry in dataset
        for j in range(n):
            center = X[j,:] - mu[i]
            all_predictions[j,i] = -(1/2) * cov_det - (1/2) * np.linalg.multi_dot([center, cov_inv, center.T]) + prior_log
    # take index with maximum value for each row and add 1 to get correct class       
    for i in range(n):
        prediction[i] = np.argmax(all_predictions[i,:]) + 1
    
    return prediction

def draw_distribution(X,y):
    for i in range(X.shape[0]):
        Xtemp = X[y == (i + 1)]
        plt.plot(Xtemp[:,0],Xtemp[:,1],'.',markersize=10)
    plt.show()

def draw_confusion_matrix(y, y_hat):
    confusion_matrix = pd.crosstab(y, y_hat,rownames=['y_truth'],colnames=['y_pred']) 
    print("Confusion Matrix:")
    print(confusion_matrix)
    print()

def draw_decision_boundary(X, y, mu, cov, prior, K):
    # for the decision boundaries we need to classify the whole grid in the figure based on the model
    # define bounds of the grid plus/minus a little offset
    min_1, max_1 = X[:, 0].min() - 1, X[:, 0].max() + 1
    min_2, max_2 = X[:, 1].min() - 1, X[:, 1].max() + 1
    # define the x and y scale
    x_1_grid = np.arange(min_1, max_1, 0.1)
    x_2_grid = np.arange(min_2, max_2, 0.1)
    # create meshgrid for given mins/maxs
    xx, yy = np.meshgrid(x_1_grid, x_2_grid)
    # flatten each grid to a vector
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
    # horizontal stack vectors to create x1,x2 input for the model
    grid = np.hstack((r1,r2))
    # make predictions for the grid
    y_hat_new = predict(grid,mu,cov,prior,K)
    # reshape the predictions back into a grid
    Z = y_hat_new.reshape(xx.shape)

    # draw decision boundary plot
    fig, ax = plt.subplots()
    ax.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.5) 
    # for contours on edges use next line instead
    # ax.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.5)
    ax.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.Paired, s=20)
    ax.plot(X[y_hat != y, 0], X[y_hat != y, 1], "ko", markersize = 12, fillstyle = "none")  
    plt.show()

if __name__ == "__main__":
    
    # generate data points from multivariate gaussian distribution
    X,y = generateData()
    
    # uncomment next line to draw distribution and check generated data
    # draw_distribution(X,y)

    # get number of classes, dimensions/features and samples
    K = np.max(y)
    N, D = X.shape

    # estimate mean, covariances and priors 
    sample_means, sample_covariances, class_priors = fit(X,y,K,D)
    
    # predict labels for training data
    y_hat = predict(X=X,mu=sample_means,cov=sample_covariances,prior=class_priors,k=K)  
    
    # draw confusion matrix
    draw_confusion_matrix(y,y_hat)
    
    # draw decision boundary
    draw_decision_boundary(X=X,y=y,mu=sample_means,cov=sample_covariances,prior=class_priors,K=K)