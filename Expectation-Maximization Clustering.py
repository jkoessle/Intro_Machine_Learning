import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as spa
from scipy.stats import multivariate_normal

DATA = ""
INITIAL_CENTROIDS = ""


def get_cov(X,N):
    # calculate covariance matrix
    return np.dot(X.T,X) / N


def get_memberships(centroids, X):
    # calculate distances between centroids and data points
    D = spa.distance_matrix(centroids, X)
    # find the nearest centroid for each data point
    memberships = np.argmin(D, axis = 0)
    return memberships


def EM(X,init,iter=100):       
    N,D = X.shape
                
    """Initialization step"""
    mean = init.copy()
    initial_member = get_memberships(mean,X)
    prior = np.asarray(np.bincount(initial_member)/len(initial_member))
    cov = np.asarray([get_cov(X[i==initial_member],N) for i in range(len(init))])
    
    # perform n iterations of EM algorithm
    for _ in range(iter):               
        """E step"""
        values = np.zeros((len(X),len(cov)))

        # calculate values for each class
        for mu,co,p,v in zip(mean,cov,prior,range(len(values[0]))):
            norm = multivariate_normal(mean=mu,cov=co)
            values[:,v] = p*norm.pdf(X)/np.sum([p_c*multivariate_normal(mean=mu_c,cov=cov_c).pdf(X) for p_c,mu_c,cov_c in zip(prior,mean,cov)],axis=0)

        """M step"""
        # calculate new mean vector, covariance matrices and priors
        for c in range(len(values[0])):
            m_c = np.sum(values[:,c],axis=0)
            mu_c = np.sum(X*values[:,c].reshape(len(X),1),axis=0) / m_c
            
            mean[c] = mu_c
            cov[c] = ((1/m_c)*np.dot((np.array(values[:,c]).reshape(len(X),1)*(X-mu_c)).T,(X-mu_c)))
            prior[c] = m_c/np.sum(values)
        
    return mean,values,cov,prior
    

def draw_cluster(X,EM_means,EM_values,cov,priors): 
    cluster_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#b15928",
                               "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6", "#ffff99"])
    
    """Draw clusters resulting from EM"""
    EM_results = np.argmax(EM_values,axis=1)
    for c in range(len(EM_means)):
            plt.plot(X[EM_results == c, 0], X[EM_results == c, 1], ".", markersize = 10,
                     color = cluster_colors[c])
    
    """Initialize helpers for contours"""
    intervals = 1000
    ys = np.linspace(-8,8,intervals)
    X_new, Y = np.meshgrid(ys, ys)
    _ys = np.vstack([X_new.ravel(), Y.ravel()]).T
    z = np.zeros(len(_ys))

    """Draw contours based on EM densities"""
    c = 0
    for pi, mu, cov in zip(priors, EM_means, cov):
        z = pi*multivariate_normal(mu, cov).pdf(_ys)
        z = z.reshape((intervals, intervals))
        plt.contour(X_new, Y, z, [0.005],colors=cluster_colors[c])
        c += 1
  
    """Initialize params of original Gaussian densities"""
    priors = np.asarray([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.2])
    means = np.asarray([[5.0,5.0],[-5.0,5.0],[-5.0,-5.0],[5.0,-5.0],
                        [5.0,0.0],[0.0,5.0],[-5.0,0.0],[0.0,-5.0],[0.0,0.0]])
    cov = np.asarray([[[0.8,-0.6],[-0.6,0.8]],[[0.8,0.6],[0.6,0.8]],[[0.8,-0.6],[-0.6,0.8]],
                      [[0.8,0.6],[0.6,0.8]],[[0.2,0.0],[0.0,1.2]],[[1.2,0.0],[0.0,0.2]],[[0.2,0.0],[0.0,1.2]],
                      [[1.2,0.0],[0.0,0.2]],[[1.6,0.0],[0.0,1.6]]])

    """Draw contours based on original densities"""
    z = np.zeros(len(_ys))
    for pi, mu, cov in zip(priors, means, cov):
        z += pi*multivariate_normal(mu, cov).pdf(_ys)

    z = z.reshape((intervals, intervals))
    plt.contour(X_new,Y,z,[0.005],linestyles="dashed")

    """Set axes and labels"""       
    ax = plt.gca()
    ax.set_ylim([-8.0, 8.0])
    ax.set_xlim([-8.0, 8.0])
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()
    

if __name__ == "__main__":
    # read data into memory
    X = np.genfromtxt(DATA, delimiter = ",")
    init = np.genfromtxt(INITIAL_CENTROIDS, delimiter = ",")
    
    # calculate EM for 100 iterations
    means,values,cov,priors = EM(X,init,iter=100)
    
    print("EM means after 100 iterations:")
    print(means)
    
    # draw clusters
    draw_cluster(X,means,values,cov,priors)