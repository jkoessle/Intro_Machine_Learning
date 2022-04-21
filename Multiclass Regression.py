import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = ""
LABELS_PATH = ""

def train_test_split(data,labels,encode=True):
    K = np.max(labels)
    
    X_1,X_2,X_3,X_4,X_5 = np.array_split(data,5)

    y_1,y_2,y_3,y_4,y_5 = np.array_split(labels,5)

    X_train = np.concatenate((X_1[:25,:],X_2[:25,:],X_3[:25,:],X_4[:25,:],X_5[:25,:]))

    X_test = np.concatenate((X_1[25:,:],X_2[25:,:],X_3[25:,:],X_4[25:,:],X_5[25:,:]))

    y_train = np.concatenate((y_1[:25],y_2[:25],y_3[:25],y_4[:25],y_5[:25]))

    y_test = np.concatenate((y_1[25:],y_2[25:],y_3[25:],y_4[25:],y_5[25:]))
    
    if encode:
        # one-of-K encoding for train labels
        N_train = y_train.size
        Y_train = np.zeros((N_train, K)).astype(int)
        Y_train[range(N_train), y_train - 1] = 1
        
        # one-of-K encoding for test labels
        N_test = y_test.size
        Y_test = np.zeros((N_test, K)).astype(int)
        Y_test[range(N_test), y_test - 1] = 1
        
        return X_train,X_test,Y_train,Y_test
    else:
        return y_train,y_test

def sigmoid(X, w, w_0):
    # return sigmoid function
    return (1 / (1 + np.exp(-(np.dot(X,w) + w_0))))

def gradient_w(X, y, y_hat):
    # return partial derivative of sigmoid function w.r.t. w
    return (np.asarray([np.dot(((y_hat[:,k] - y[:,k]) * y_hat[:,k] * (1 - y_hat[:,k])), X) for k in range(y.shape[1])]).transpose())   

def gradient_w_0(y, y_hat):
    # return partial derivative of sigmoid function w.r.t. w_0
    return (np.sum((y_hat - y) * y_hat * (1 - y_hat), axis = 0))

def optimize(X,y,eta,epsilon,w,w_0,verbose=False):  
    
    iteration = 0
    objective_values = []
    
    while True:
        
        # sigmoid error function
        y_hat = sigmoid(X, w, w_0)
        objective_value = 0.5 * (np.sum((y - y_hat)**2))
        
        # collect objective values for convergence figure
        objective_values = np.append(objective_values,objective_value)
        
        w_old = w
        w_0_old = w_0

        # adapt new weights w.r.t. gradient and learning rate
        w = w - eta * gradient_w(X, y, y_hat)
        w_0 = w_0 - eta * gradient_w_0(y, y_hat)

        iteration += 1
        
        # if verbose = True, prints out current objective value for each iteration
        if verbose:
            print("Epoch: " + str(iteration) + ", Objective Function Value: " + str(objective_value))

        # convergence criteria
        if (np.sqrt(np.sum((w_0 - w_0_old))**2 + np.sum((w - w_old)**2)) < epsilon):
            break
        
    print("Weights w:")
    print(w)
    print()
    print("Weights w_0:")
    print(w_0)    
        
    return w,w_0,iteration,objective_values

def predict(X, w, w_0):
    # compute predictions based on optimized parameters
    y_hat = sigmoid(X, w, w_0)    

    # pick class with highest value
    predictions = np.argmax(y_hat, axis = 1) + 1

    return predictions

def draw_confusion_matrix(y, y_hat):
    # print confusion matrix
    confusion_matrix = pd.crosstab(y_hat,y,rownames=['y_pred'],colnames=['y_truth']) 
    print("Confusion Matrix:")
    print(confusion_matrix)
    print()
    
def plot_convergence(iterations,values):
    # plot objective function during iterations
    plt.figure(figsize = (8, 4))
    plt.plot(range(1, iterations + 1), values, "k-")
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.show()
    
if __name__ == "__main__":
    # read data into memory
    image_data = np.genfromtxt(DATA_PATH, delimiter = ",")#.astype(int)
    image_labels = np.genfromtxt(LABELS_PATH, delimiter = ",").astype(int)
    
    # split train/test data
    X_train, X_test, y_train, y_test = train_test_split(image_data,image_labels)
    
    # set learning parameters
    eta = 0.001
    epsilon = 0.001

    # get number of classes
    K = y_train.shape[1]
    
    # get number of dimensions
    D = X_train.shape[1]
    
    # randomly initalize w and w0; random seed 42 because that is the answer to everything
    np.random.seed(42)
    w = np.random.uniform(low = -0.01, high = 0.01, size = (D, K))
    w_0 = np.random.uniform(low = -0.01, high = 0.01, size = (1, K))
    
    # optimize weights
    w_train,w_0_train,iterations_train,values_train = optimize(X_train,y_train,eta,epsilon,w,w_0)
    
    # predict labels for train and test set
    y_hat_train = predict(X_train, w_train, w_0_train)
    y_hat_test = predict(X_test, w_train, w_0_train)
    
    # get 1-dimensional y-labels
    Y_train,Y_test = train_test_split(image_data,image_labels,encode=False)
    
    # print confusion matrices
    draw_confusion_matrix(Y_train, y_hat_train)
    
    draw_confusion_matrix(Y_test, y_hat_test) 
    
    # plot convergence
    plot_convergence(iterations_train,values_train)    