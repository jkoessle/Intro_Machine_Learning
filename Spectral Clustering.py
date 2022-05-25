import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as spa
import scipy.sparse.csgraph as cs
import networkx as nx

from scipy.sparse.linalg import eigs

DATA_PATH = ""


def get_B(X,delta):
    # compute distance matrix
    norm = spa.distance_matrix(X,X)
    
    # compute B matrix
    B = ((norm < delta) & (norm != 0)).astype(int)
    
    return B


def draw_network_graph(X,adjacency_matrix):
    graph = nx.Graph()
    
    for node in range(len(X)):
        graph.add_node(node)
        
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    
    graph.add_edges_from(edges)

    fig, ax = plt.subplots()
    nx.draw(graph, [(x,y) for x,y in X], node_size=10,node_color='black',edge_color='grey',ax=ax)
    plt.axis("on")
    ax = plt.gca()
    ax.set_ylim([-8.0, 8.0])
    ax.set_xlim([-8.0, 8.0])
    plt.xlabel("x1")
    plt.ylabel("x2")
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.show()


def laplace(B,X):
    N,D = B.shape
    
    # compute D matrix
    D = np.zeros((N,D))
    for n in range(N):
        D[n,n] = np.sum(B[n,:])
    
    # generate identity matrix
    I = np.identity(len(B))
    
    # inverse D matrix for Laplace 
    D_inv_sqrt = np.linalg.inv(np.sqrt(D))
    
    # SciPy Laplace matrix 
    # L = cs.laplacian(B,normed=True)
    
    # compute Laplace matrix as defined in lecture
    L = I - np.dot(D_inv_sqrt, B).dot(D_inv_sqrt)
    
    # get six smallest eigenvectors
    _,v = eigs(L,k=6,which="SM")
    
    # cast values to real format to discard imaginary part
    vectors = v.real
    
    # select five smallest eigenvectors
    # discard smallest one because eigenvalue = 0
    Z = vectors[:,1:6]
    
    return L,Z


def draw_cluster(centroids, memberships, X, K):
    cluster_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#b15928",
                               "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6", "#ffff99"])
    if memberships is None:
        plt.plot(X[:,0], X[:,1], ".", markersize = 10, color = "black")
    else:
        for c in range(K):
            plt.plot(X[memberships == c, 0], X[memberships == c, 1], ".", markersize = 10,
                     color = cluster_colors[c])        
    
    for c in range(K):
        plt.plot(centroids[c, 0], centroids[c, 1], "s", markersize = 12, 
                 markerfacecolor = cluster_colors[c], markeredgecolor = "black")
    ax = plt.gca()
    ax.set_ylim([-8.0, 8.0])
    ax.set_xlim([-8.0, 8.0])
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()
    
    
def get_member(points, centroids):
    # compute distances and return memberships
    distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)


def get_centroids(points, closest, centroids):
    # compute nearest centroids
    return np.array([points[closest==k].mean(axis=0) for k in range(centroids.shape[0])])


def k_M(matrix,X):
    # initialize centroids and memberships
    mask = np.asarray((242,528,570,590,648,667,774,891,955))
    centroids = matrix[mask,:]
    member = np.zeros(matrix.shape[0])
    
    # Break condition
    cond = True
    
    while cond:
        old_member = member.copy()
        
        # get new memberships and centroids
        member = get_member(matrix, centroids)
        centroids = get_centroids(matrix, member, centroids)
        
        # if memberships dont change - stop
        if np.all(old_member == member):
            cond = False
            
            # get centroids for dataset X
            X_centroids = get_centroids(X,member,centroids)
            
            # draw cluster for dataset X
            draw_cluster(X_centroids,member,X,centroids.shape[0])
             
    
if __name__ == "__main__":
    # read data into memory
    X = np.genfromtxt(DATA_PATH, delimiter = ",")
    
    # get B matrix
    B = get_B(X,delta=2.0)
    
    # draw network graph
    draw_network_graph(X,B)
    
    # get L and Z matrix
    L,Z = laplace(B,X)
    
    print("L Matrix:")
    print(L[0:5, 0:5])
    print("Z Matrix:")
    print(Z[0:5, 0:5])
    
    # perform k-Means clustering
    k_M(Z,X)