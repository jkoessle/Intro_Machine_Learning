import numpy as np
import matplotlib.pyplot as plt

TRAIN_PATH = ""
TEST_PATH = ""

def rmse(X,y_hat):
    y_ax = X[:,1]
    
    square_sum = np.sum((y_ax - y_hat)**2)
    
    rmse = np.sqrt(square_sum / len(y_ax))
    
    return rmse
    
def train_tree(x,P):
    X = x[:,0]
    y = x[:,1]
    
    N = X.shape[0]
    
    # create necessary data structures
    node_indices = {}
    is_terminal = {}
    need_split = {}
    means = {}
    
    node_features = {}
    node_splits = {}

    # put all training instances into the root node
    node_indices[1] = np.array(range(N))

    is_terminal[1] = False
    need_split[1] = True
    
    # learning algorithm
    while True:
        # find nodes that need splitting
        split_nodes = [key for key, value in need_split.items()
                    if value == True]

        # check whether we reach all terminal nodes
        if len(split_nodes) == 0:
            break
        # find best split positions for all nodes
        for split_node in split_nodes:
            

            data_indices = node_indices[split_node]

            need_split[split_node] = False

            
            node_mean = np.mean(y[data_indices])

            if len(X[data_indices]) <= P:
                is_terminal[split_node] = True
                means[split_node] = node_mean
            else:
                is_terminal[split_node] = False

                best_score = np.repeat(0.0, 1)
              
                best_split = np.repeat(0.0, 1)
                unique_values = np.sort(np.unique(X[data_indices]))
             
                split_positions = (unique_values[1:len(unique_values)] + \
                                    unique_values[0:(len(unique_values) - 1)]) / 2
        
                split_scores = np.repeat(0.0, len(split_positions))
                for s in range(len(split_positions)):
                    error = 0
                    
                    left_indices = data_indices[X[data_indices] > split_positions[s]]

                    right_indices = data_indices[X[data_indices] <= split_positions[s]]
                    
                    indices = np.concatenate((left_indices,right_indices), axis=0)
                    
                    if len(left_indices) > 0:
                        error = tree_error(y, left_indices, error)
                        
                    if len(right_indices) > 0:
                        error = tree_error(y, right_indices, error)
                    
                    split_scores[s] = error / (len(indices))    
                
                if len(np.unique(y[data_indices])) == 1:
                    is_terminal[split_node] = True
                    means[split_node] = node_mean
                    continue
                    
                best_score = np.min(split_scores)

                best_split = split_positions[np.argmin(split_scores)]
                
                node_features[split_node] = best_score
                
                node_splits[split_node] = best_split
                
                # create left node using the selected split
                left_indices = data_indices[X[data_indices] > best_split]
                node_indices[2 * split_node] = left_indices
                is_terminal[2 * split_node] = False
                need_split[2 * split_node] = True
        
                # create right node using the selected split
                right_indices = data_indices[X[data_indices] <= best_split]
                node_indices[2 * split_node + 1] = right_indices
                is_terminal[2 * split_node + 1] = False
                need_split[2 * split_node + 1] = True     
    
    return node_splits, means, is_terminal

def tree_error(y, index, error): 
    mean = np.mean(y[index])
    sum_square = np.sum((y[index] - mean)**2)
    
    error = error + sum_square
    
    return error

def predict(X, split_nodes, means, terminal_nodes):
    N = X.shape[0]
    y_hat = np.zeros(N)
    for i in range(N):
        index = 1
        while True:
            if terminal_nodes[index] == True:
                y_hat[i] = means[index]
                break
            else:
                if X[i] > split_nodes[index]:
                    index = index * 2
                else:
                    index = index * 2 + 1     
    return y_hat
    
def draw_decision_tree(X, data_interval,y_hat,color):
    plt.figure(figsize = (10, 6))
    plt.plot(X[:,0], X[:,1], color, markersize = 10)
    plt.plot(data_interval, y_hat, "k-")
    ax = plt.gca()
    ax.set_ylim([-1.0, 2.0])
    plt.xlabel("Time (sec)")
    plt.ylabel("Signal (millivolt)")
    plt.show()
    
def draw_rmse(rmse_train,rmse_test,p_values):
    plt.figure(figsize = (10, 6))
    plt.plot(p_values, rmse_train, "-bo", label="training")
    plt.legend()
    plt.plot(p_values, rmse_test, "-ro", label="test")
    plt.xlabel("Pre-pruning size (P)")
    plt.ylabel("RMSE")
    plt.legend()
    # ax = plt.gca()
    # ax.set_ylim([-1.0, 2.0])
    plt.show()
    
if __name__ == "__main__":
    # read data into memory
    train_data = np.genfromtxt(TRAIN_PATH, delimiter = ",")
    test_data = np.genfromtxt(TEST_PATH, delimiter = ",")
    
    P = 30
    
    splits, means, terminal = train_tree(x=train_data, P=P)
    
    # print(terminal)
    
    y_hat_train = predict(train_data[:,0], splits, means, terminal)
    y_hat_test = predict(test_data[:,0], splits, means, terminal)
    
    rmse_train = rmse(train_data, y_hat_train)
    rmse_test = rmse(test_data, y_hat_test)
    
    print(f"RMSE on training set is {rmse_train} when P is {P}")
    print(f"RMSE on training set is {rmse_test} when P is {P}")
    
    # draw decision tree for training dataset
    data_interval = np.arange(0, 2.0, 0.001)
    y_hat_draw = predict(data_interval, splits, means, terminal)
    draw_decision_tree(train_data,data_interval,y_hat_draw,color="b.")
    draw_decision_tree(test_data,data_interval,y_hat_draw,color="r.")
    
    list_y_hat_train = []
    list_y_hat_test = []
    count = 0
    for p in range(10,51,5):
        splits,means,terminal = train_tree(x=train_data, P=p)
        
        list_y_hat_train.append(predict(train_data[:,0],splits,means,terminal))
        list_y_hat_test.append(predict(test_data[:,0],splits,means,terminal))

    p_values = np.arange(10,51,5)  
        
    list_rmse_train = []    
    list_rmse_test = []
    for i in range(len(p_values)):     
        list_rmse_train.append(rmse(train_data,list_y_hat_train[i]))
        list_rmse_test.append(rmse(test_data,list_y_hat_test[i]))
    
    draw_rmse(list_rmse_train,list_rmse_test,p_values)
    
    
    
    