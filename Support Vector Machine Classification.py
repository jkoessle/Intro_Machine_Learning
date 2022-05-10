import numpy as np
import pandas as pd
import cvxopt as cvx
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def train_test_split(data,labels):
    
    X_train, X_test = np.split(data, 2)
    y_train, y_test = np.split(labels, 2)
    
    return X_train,X_test,y_train,y_test

def color_histogram(X):
    N, K = X.shape
    bins = np.arange(0,256,4)
    hist = np.zeros((N,len(bins)))
    
    for i in range(N):
        for j in range(len(bins)):
            hist[i,j] = np.count_nonzero((X[i,:] >= bins[j]) & (X[i,:] < (bins[j] + 4))) / K
        
    return hist

def hist_kernel(h_1,h_2):
    N,K = h_1.shape
    kernel = np.zeros((N,N))

    for i in range(N):
        h1_v = np.vstack(h_1[i,:])
        for j in range(N):
            h2_v = np.vstack(h_2[j,:])       
            kernel[i,j] = np.sum(np.minimum(h1_v,h2_v))

            
    return kernel

def train_kernel(kernel,y,C):
    N,K = kernel.shape
    # set learning parameter
    epsilon = 0.001
    
    yyK = np.matmul(y[:,None], y[None,:]) * kernel

    P = cvx.matrix(yyK)
    q = cvx.matrix(-np.ones((N, 1)))
    G = cvx.matrix(np.vstack((-np.eye(N), np.eye(N))))
    h = cvx.matrix(np.vstack((np.zeros((N, 1)), C * np.ones((N, 1)))))
    A = cvx.matrix(1.0 * y[None,:])
    b = cvx.matrix(0.0)
                        
    # use cvxopt library to solve QP problems
    # turn off progress
    cvx.solvers.options['show_progress'] = False
    result = cvx.solvers.qp(P, q, G, h, A, b)
    alpha = np.reshape(result["x"], N)
    alpha[alpha < C * epsilon] = 0
    alpha[alpha > C * (1 - epsilon)] = C
    
    # find bias parameter
    support_indices, = np.where(alpha != 0)
    active_indices, = np.where(np.logical_and(alpha != 0, alpha < C))
    w0 = np.mean(y_train[active_indices] * (1 - np.matmul(yyK[np.ix_(active_indices, support_indices)], alpha[support_indices])))
    
    return alpha, w0

def predict(kernel,y,alpha,w0):
    # calculate predictions
    f_predicted = np.matmul(kernel, y[:,None] * alpha[:,None]) + w0

    # calculate confusion matrix
    y_hat = 2 * (f_predicted > 0.0) - 1
    return y_hat

def draw_confusion_matrix(y, y_hat):
    N = len(y)
    confusion_matrix= pd.crosstab(np.reshape(y_hat, N), y,
                               rownames = ["y_predicted"], colnames = ["y_train"])
    print("Confusion Matrix:")
    print(confusion_matrix)
    print()
    
    
def draw_accuracy(accuracy_train,accuracy_test,c_values):
    plt.figure(figsize = (10, 6))
    plt.semilogx(c_values, accuracy_train, "-bo", label="training")
    plt.semilogx(c_values, accuracy_test, "-ro", label="test")
    plt.xlabel("Regularization parameter (C)")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()        
    
    
if __name__ == "__main__":
    # read data into memory
    image_data = np.genfromtxt("hw06_data_set_images.csv", delimiter = ",")
    image_labels = np.genfromtxt("hw06_data_set_labels.csv", delimiter = ",").astype(int)
    
    # split train/test data
    X_train, X_test, y_train, y_test = train_test_split(image_data,image_labels)
    
    # generate histograms for train and test data
    H_train = color_histogram(X_train)
    H_test = color_histogram(X_test)
    
    # print first entries of train and test histograms
    print("First entries for H_train:")
    print(H_train[0:5,0:5])
    print("First entries for H_test:")
    print(H_test[0:5,0:5])
    
    # generate kernels for train and test data
    K_train = hist_kernel(H_train,H_train)
    K_test = hist_kernel(H_test,H_train)
    
    # Save/Load calculated kernels
    # np.savetxt("K_train.csv", K_train, delimiter=",")
    # np.savetxt("K_test.csv", K_test, delimiter=",")
    
    # load saved kernels
    # K_train = np.genfromtxt("K_train.csv", delimiter = ",")
    # K_test = np.genfromtxt("K_test.csv", delimiter = ",")
    
    # print first entries of train and test kernels
    print("First entries for K_train:")
    print(K_train[0:5,0:5])
    print("First entries for K_test:")
    print(K_test[0:5,0:5])
    
    # train kernel on train data
    alpha, w0 = train_kernel(K_train,y_train,C=10)
    
    # predict values for train and test data 
    y_hat_train = predict(K_train, y_train, alpha, w0)
    y_hat_test = predict(K_test, y_train, alpha, w0)
    
    # draw confusion matrices
    draw_confusion_matrix(y_train,y_hat_train)
    draw_confusion_matrix(y_test,y_hat_test)
    
    # C values
    C_power = [10**-1,10**-0.5,10**0,10**0.5,10**1,10**1.5,10**2,10**2.5,10**3]
    
    # lists for y_hat values
    list_y_hat_train = []
    list_y_hat_test = []
    
    # lists for accuracy values
    list_accuracy_train = []    
    list_accuracy_test = []
    
    # predict y_hat for range of C values
    for i in range(len(C_power)):
        alpha, w0 = train_kernel(K_train,y_train,C=C_power[i])
        list_y_hat_train.append(predict(K_train, y_train, alpha, w0))
        list_y_hat_test.append(predict(K_test, y_train, alpha, w0))

    # compute accuracy for y_hat values
    for i in range(len(C_power)):     
        list_accuracy_train.append(accuracy_score(y_train,list_y_hat_train[i]))
        list_accuracy_test.append(accuracy_score(y_test,list_y_hat_test[i]))
        
    # draw accuracy for train and test data    
    draw_accuracy(list_accuracy_train, list_accuracy_test, C_power)
    
    
    
    
    
    
    
    
