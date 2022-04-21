import numpy as np
import matplotlib.pyplot as plt
import math

TRAIN_PATH = ""
TEST_PATH = ""

def rmse(X,y_hat,model):
    if model == "Kernel Smoother":
        bin_width = 0.02
    else:
        bin_width = 0.1

    y_ax = X[:,1]
    
    rmse = math.sqrt(np.square(np.subtract(y_ax,y_hat)).mean())
    
    print(f"{model} => RMSE is {rmse} when h is {bin_width}")
    
def repeat_array(x,y_hat):
    bin_width = 0.1
    origin = 0.0
    max = np.max(x)
    
    # arrange borders
    left_border = np.arange(origin, max, bin_width)
    right_border = np.arange(origin + bin_width, max + bin_width, bin_width)
    
    # repeat y_hat values for all data points
    repeat_array = np.repeat(y_hat,[np.count_nonzero((left_border[b] < x) & (x <= right_border[b])) for b in range(len(y_hat))])
    
    return repeat_array

def downsample_array(X,y_hat,data_interval):
    N = X.shape[0]
    x = X[:,0]
    
    # create sorter and find matching indices between data interval and actual data points
    sorter = np.argsort(data_interval)
    sorter = sorter[np.searchsorted(data_interval, x, sorter=sorter)]
    
    # assert that sorter is of length = N
    assert len(sorter) == N
    
    # create new array with entries for actual data points
    y_downsampled = y_hat[sorter]
    
    return y_downsampled

def train_regressogram(X):
    bin_width = 0.1
    origin = 0.0
    max = np.max(X[:,0])
    
    x = X[:,0]
    y = X[:,1]
    
    # arrange borders
    left_border = np.arange(origin, max, bin_width)
    right_border = np.arange(origin + bin_width, max + bin_width, bin_width)

    # calculate y_hat based on bin width
    y_hat = np.asarray([np.sum(x[(left_border[b] < y) & (y <= right_border[b])]) 
                        / np.count_nonzero((left_border[b] < y) & (y <= right_border[b])) for b in range(len(left_border))])
    
    return y_hat

def draw_regressogram(X,y_hat,color):
    bin_width = 0.1
    origin = 0.0
    max = np.max(X[:,0])
    
    # arrange borders
    left_border = np.arange(origin, max, bin_width)
    right_border = np.arange(origin + bin_width, max + bin_width, bin_width)
    
    plt.figure(figsize = (10, 6))
    plt.plot(X[:,0], X[:,1], color, markersize = 10)
    for b in range(len(left_border)):
        plt.plot([left_border[b], right_border[b]], [y_hat[b], y_hat[b]], "k-")
    for b in range(len(left_border) - 1):
        plt.plot([right_border[b], right_border[b]], [y_hat[b], y_hat[b + 1]], "k-") 
    
    ax = plt.gca()
    ax.set_ylim([-1.0, 2.0])
    plt.show()  

def train_running_mean(X,data_interval):
    bin_width = 0.1
    x_ax = X[:,0]
    y_ax = X[:,1]
    
    # calculate y_hat for every datapoint in data interval
    y_hat = np.asarray([np.sum(y_ax[(np.abs(x - x_ax) / bin_width) <= 0.5]) / np.count_nonzero(y_ax[(np.abs(x - x_ax) / bin_width) <= 0.5]) for x in data_interval])
          
    return y_hat

def draw_running_mean(X,y_hat,color,data_interval):   
    plt.figure(figsize = (10, 6))
    plt.plot(X[:,0], X[:,1], color, markersize = 10)
    plt.plot(data_interval, y_hat, "k-")
    ax = plt.gca()
    ax.set_ylim([-1.0, 2.0])
    plt.show()
    
def train_kernel_smoother(X,data_interval):
    bin_width = 0.02
    N = X.shape[0]
    
    x_ax = X[:,0]
    y_ax = X[:,1]
    
    # calculate y_hat for every datapoint in data interval
    y_hat = np.asarray([np.sum(1.0 / np.sqrt(2 * math.pi) * np.exp(-0.5 * (x - x_ax)**2 / bin_width**2) * y_ax) / 
                        np.sum(1.0 / np.sqrt(2 * math.pi) * np.exp(-0.5 * (x - x_ax)**2 / bin_width**2)) for x in data_interval])
    return y_hat

def draw_kernel_smoother(X,y_hat,color,data_interval):
    plt.figure(figsize = (10, 6))
    plt.plot(X[:,0], X[:,1], color, markersize = 10)
    plt.plot(data_interval, y_hat, "k-")
    ax = plt.gca()
    ax.set_ylim([-1.0, 2.0])
    plt.show()

if __name__ == "__main__":
    # read data into memory
    train_data = np.genfromtxt(TRAIN_PATH, delimiter = ",")
    test_data = np.genfromtxt(TEST_PATH, delimiter = ",")
    
    # learn regressogram
    y_regressogram = train_regressogram(test_data)
    
    # draw regressograms for train and test data
    draw_regressogram(train_data, y_regressogram, "b.")
    draw_regressogram(test_data, y_regressogram, "r.")
    
    # reshape data
    y_hat_reshape = repeat_array(test_data[:,0],y_regressogram)
    
    # compute RMSE for regressogram
    rmse(test_data,y_hat_reshape,"Regressogram")
    
    # create data for drawing
    min = np.min(train_data[:,0])
    max = np.max(train_data[:,0])
    data_interval = np.linspace(min, max, 2000)
    
    # learn running mean smoother
    y_rm = train_running_mean(test_data,data_interval)
    
    # draw running mean smoother for train and test data
    draw_running_mean(train_data, y_rm, "b.", data_interval)
    draw_running_mean(test_data, y_rm, "r.", data_interval)
    
    # downsample y_hat for RMSE
    y_rm_down = downsample_array(test_data,y_rm,data_interval)
    
    # compute RSME for running mean smoother
    rmse(test_data,y_rm_down,"Running Mean Smoother")
    
    # learn kernel smoother
    y_kernel = train_kernel_smoother(train_data, data_interval)
    
    # draw kernel smoother for train and test data
    draw_kernel_smoother(train_data, y_kernel, "b.", data_interval)
    draw_kernel_smoother(test_data, y_kernel, "r.", data_interval)
    
    # downsample y_hat for RMSE
    y_kernel_down = downsample_array(test_data,y_kernel,data_interval)
    
    # compute RMSE for kernel smoother
    rmse(test_data,y_kernel_down,"Kernel Smoother")