import matplotlib.pyplot as plt
import numpy as np
import csv
import scipy.stats as stats
import pandas as pd
import scipy.spatial.distance as distance
chi2 = stats.chi2
import scipy as sp
from sklearn.metrics import confusion_matrix
plt.style.use('ggplot')

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 13
import sklearn.metrics as metrics
from numpy import genfromtxt
from scipy.stats import multivariate_normal
from sklearn.metrics import f1_score

train_data = pd.read_csv('Gussian_model_dataset/modelica_trainset.csv').astype("float")
test_data = pd.read_csv('Gussian_model_dataset/modelica_testset.csv').astype("float")
columnsName_train=['P','PE','T']
columnsName_test=['P','PE','T','attack_state']

train_data.columns=columnsName_train
test_data.columns=columnsName_test

labels= test_data[['attack_state']][:30000]


#train_data=train_data.drop(['time'], axis=1)
Test_data=test_data.drop(['attack_state'], axis=1)
#tr = list(train_data)
#cv = list(test_data)

Train_data = train_data.values
Crossval_data = Test_data.values[:30000,:]

def read_dataset(filePath, delimiter=','):
    return genfromtxt(filePath, delimiter=delimiter)


def feature_normalize(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    return (dataset - mu) / sigma


def estimate_gaussian(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.cov(dataset.T)
    return mu, sigma


def multivariate_gaussian(dataset, mu, sigma):
    p = multivariate_normal(mean=mu, cov=sigma)
    return p.pdf(dataset)

def mahalanobis(x=None, Mean=None, Cov=None):
    """Compute the Mahalanobis Distance between each row of x and the data
    x    : vector or matrix of data with, say, p columns.
    data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
    cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
    """
    x_minus_mu = x - Mean
    #if not cov:
        #cov = np.cov(data.values.T)
    cov=Cov
    inv_covmat = sp.linalg.inv(cov)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    return mahal.diagonal()


def select_threshold(probs, test_data):
    best_epsilon = 0
    best_f1 = 0
    f = 0
    stepsize = (max(probs) - min(probs)) / 1000;
    epsilons = np.arange(min(probs), max(probs), stepsize)
    for epsilon in np.nditer(epsilons):
        predictions = (probs < epsilon)
        f = f1_score(test_data, predictions, average='binary')
        if f > best_f1:
            best_f1 = f
            best_epsilon = epsilon

    return best_f1, best_epsilon

def find_TP(y_true, y_pred):
    # counts the number of true positives (y_true = 1, y_pred = 1)
    return np.sum(np.logical_and(y_pred == 1, y_true == 1))
def find_FN(y_true, y_pred):
    # counts the number of false negatives (y_true = 1, y_pred = 0)
    return np.sum(np.logical_and(y_pred == 0, y_true == 1))
def find_FP(y_true, y_pred):
    # counts the number of false positives (y_true = 0, y_pred = 1)
    return np.sum(np.logical_and(y_pred == 1, y_true == 0))
def find_TN(y_true, y_pred):
    # counts the number of true negatives (y_true = 0, y_pred = 0)
    return np.sum(np.logical_and(y_pred == 0, y_true == 0))

mu, sigma = estimate_gaussian(Train_data)
print(mu,sigma)
p_mahalanobis = mahalanobis(Crossval_data,mu,sigma)
#p_multivariate = multivariate_gaussian(test_data,mu,sigma)

#print(p_multivariate)
#exit()
k=np.arange(0,0.101,0.01)
fpr_x=[]
tpr_y=[]
#print(k)
for i in range (11):

    threshold=chi2.ppf((1-k[i]), df=2)
    print(threshold)
    output = (p_mahalanobis >= threshold).astype('int')
    #print(train_data.predicted_RF.values)
    #fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    #confusion_matrix(test_data.ix[:,0], test_data.predicted_RF.values)
    fp=find_FP(np.array(labels),output)
    print(fp)
    tn=find_TN(np.array(labels),output)
    tp=find_TP(np.array(labels),output)
    fn=find_FN(np.array(labels),output)
    fpr=fp/(fp+tn)

    tpr=tp/(tp+fn)
    fpr_x.append(fpr)
    tpr_y.append(tpr)
print(fpr_x)
print(tpr_y)
fig, ax = plt.subplots(figsize=(4, 1))
ax.plot(fpr_x, tpr_y, 'g-.', label='ROC')
legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')
yticks=np.linspace(-0.7,0.2,6)
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.grid(True)
legend = ax.legend(loc='upper left', shadow=False, fontsize='x-large')
plt.ylabel('TPR', fontsize='x-large',fontweight='bold')
ax.set_xlabel('FPR', fontsize='x-large',fontweight='bold')
plt.show()
#p = multivariate_gaussian(Train_data[:3000,:],mu,sigma)
'''p_cv = multivariate_gaussian(Crossval_data,mu,sigma)
fscore, ep = select_threshold(p_cv,labels)
print(fscore, ep)

#selecting outlier datapoints
outliers = np.asarray(np.where(p < ep))

plt.figure(1)
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')
plt.title('Datapoints of throughput vs latency')
plt.plot(train_data[:,0], train_data[:,1],'b+')
plt.show()

plt.figure(2)
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')
plt.title('Detection of Outliers')
plt.plot(train_data[:,0],train_data[:,1],'bx')
plt.plot(train_data[outliers,0],train_data[outliers,1],'ro')
plt.show()'''
