import numpy as np
import pandas as pd

DATA_PATH = ""
LABELS_PATH = ""

def train_test_split(data,labels):
    
    X_1,X_2,X_3,X_4,X_5 = np.array_split(data,5)

    y_1,y_2,y_3,y_4,y_5 = np.array_split(labels,5)

    X_train = np.concatenate((X_1[:25,:],X_2[:25,:],X_3[:25,:],X_4[:25,:],X_5[:25,:]))

    X_test = np.concatenate((X_1[25:,:],X_2[25:,:],X_3[25:,:],X_4[25:,:],X_5[25:,:]))

    y_train = np.concatenate((y_1[:25],y_2[:25],y_3[:25],y_4[:25],y_5[:25]))

    y_test = np.concatenate((y_1[25:],y_2[25:],y_3[25:],y_4[25:],y_5[25:]))
    
    return X_train,X_test,y_train,y_test

def fit(X,y):
    N,D = X.shape
    C = np.max(y)
    
    pcd = np.zeros((C,D))
    
    # compute conditional probabilities
    for c in range(C):
        Xtemp=X[(y==(c+1))]
        for d in range(D):
            counts = np.sum(Xtemp[:,d])
            pcd[c,d] = counts
        
        pcd[c,:] /= np.count_nonzero(y[y==(c+1)])
         
    print("PCD:")
    print(pcd)
    print()        
    
    # compute prior probabilities
    class_priors = [np.mean(y == (i + 1)) for i in range(C)]
    print("Sample Priors:")
    print(class_priors)
    print()
    
    return pcd,class_priors
    
def predict(X,pcd,priors,K):
    N,D = X.shape
    
    all_predictions = np.zeros((N,K))
    prediction = np.zeros(N)
    
    # ignore division by zero runtime warning
    np.seterr(divide='ignore')
    
    # compute logs for priors and pcd's
    log_prior = np.log(priors)
    log_pcd = np.log(pcd)
    log_pcd_minus = np.log(1-pcd)
    
    # turn warning for division by zero back on
    np.seterr(divide='warn')
    
    # compute probabilities for every value for every feature
    for k in range(K):        
        for n in range(N):
            logsum = 0
            for d in range(D):
                if X[n,d] == 1:
                    logsum += log_pcd[k,d]
                else:
                    logsum += log_pcd_minus[k,d]
                    
            all_predictions[n,k] = log_prior[k] + logsum

    # select class with highest probability
    for i in range(N):
        prediction[i] = np.argmax(all_predictions[i,:]) + 1
    
    # return only predicted labels        
    return prediction

def draw_confusion_matrix(y, y_hat):
    # print confusion matrix
    confusion_matrix = pd.crosstab(y_hat,y,rownames=['y_pred'],colnames=['y_truth']) 
    print("Confusion Matrix:")
    print(confusion_matrix)
    print()
    
    
    
if __name__ == "__main__":
    # read data into memory
    image_data = np.genfromtxt(DATA_PATH, delimiter = ",")#.astype(int)
    image_labels = np.genfromtxt(LABELS_PATH, delimiter = ",").astype(int)
    
    # split train/test data
    X_train, X_test, y_train, y_test = train_test_split(image_data,image_labels)
    
    # fit nb estimator to train data
    sample_pcd, sample_priors = fit(X_train,y_train)
    
    # predict data points for train data
    y_hat = predict(X_train, sample_pcd, sample_priors, np.max(y_train))
    draw_confusion_matrix(y_train,y_hat)
    
    # predict data points for test data
    y_hat_test = predict(X_test, sample_pcd, sample_priors, np.max(y_test))
    draw_confusion_matrix(y_test,y_hat_test)
    
    
    
    
