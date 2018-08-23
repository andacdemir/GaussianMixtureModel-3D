import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.preprocessing import StandardScaler
from sklearn import mixture
from sklearn import svm
from sklearn.metrics import classification_report
from  sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from classification_utilities import display_cm, display_adj_cm

'''
    Reads the processed datasets (las files)
'''
def read_data():
    B15_data = pd.read_excel("Processed_LogData.xlsx", header=None, skiprows=1) 
    # converts the data frame to numpy array preserving the indices:
    B15_data = B15_data.values
    feature_vectors = B15_data[:,1:4]
    return B15_data, feature_vectors

'''
    Standard scaling of the feature vectors in the dataset
    Makes the features 0 mean and unit variance
'''
def standardize(feature_vectors):
    scaler = StandardScaler().fit(feature_vectors)
    scaled_features = scaler.transform(feature_vectors)
    return scaled_features

'''
    This returns the first and second derivatives of the BIC vector.
'''
def derivatives_calc(bic_mean_cluster): 
    der_1st = np.zeros((bic_mean_cluster.shape[0],)*1) 
    der_2nd = np.zeros((bic_mean_cluster.shape[0],)*1) 
    der_1st = np.gradient(bic_mean_cluster)    
    der_2nd = np.gradient(der_1st)
    return der_1st, der_2nd

'''
    Takes the scaled features and max_clusters which is a hypothetical number
    with a default value 30. A higher number slows down the computation.
    Computes the Bayesian Inference Criterion for different number of clusters
    Additionally, computes the first and second derivatives of them
'''
def compute_BIC(scaled_features, covariance_type='full', max_clusters=20):
    cluster_ref = range(1, max_clusters) 
    iters = 10 # number of iterations

    bic_mean_cluster = np.zeros((len(cluster_ref),)*1) 
    bic_std_cluster = np.zeros((len(cluster_ref),)*1) 

    for i in range(len(cluster_ref)):  
        print(i)
        bic = []
        for _ in range(iters):
            gmm = mixture.GaussianMixture(cluster_ref[i], covariance_type)\
                                                     .fit(scaled_features)
            bic.append(gmm.bic(scaled_features))
        bic_mean_cluster[i] = np.mean(bic)
        bic_std_cluster[i] = np.std(bic)

    der_1st, der_2nd = derivatives_calc(bic_mean_cluster)
    return bic_mean_cluster, der_1st, der_2nd   

'''
    Plots the BIC values corresponding to each cluster
    Additionally plots the first and second derivatives of the BIC curve 
'''
def plot_BIC(bic_mean_cluster, der_1st, der_2nd, max_clusters=16):
    plt.figure(figsize=(20,10))

    plt.subplot(3,2,1)
    plt.plot(bic_mean_cluster, marker='o')
    plt.xlim(1, max_clusters)
    plt.ylabel("BIC")

    plt.subplot(3,2,3)
    plt.plot(der_1st, marker='o')
    plt.xlim(1, max_clusters)
    plt.ylabel("1st derivative")

    plt.subplot(3,2,5)
    plt.plot(der_2nd, marker='o')
    plt.xlim(1, max_clusters)
    plt.ylabel("2nd derivative")

    plt.show()

'''
    Confusion matrix should be a diagonal matrix.
    Computes the accuracy by dividing the sum of diagonal elements of the 
    confusion matrix to the sum of all elements.
'''
def get_accuracy(confusion_matrix):
    total_correct = 0.
    num_classes = confusion_matrix.shape[0]
    for i in np.arange(num_classes):
        total_correct += confusion_matrix[i][i]
    acc = total_correct/sum(sum(confusion_matrix))
    
    return acc

def validate_GMM(scaled_features, num_clusters, covariance_type='full'):
    clf = mixture.GaussianMixture(num_clusters, covariance_type)\
                                             .fit(scaled_features)
    y_pred = clf.predict(scaled_features)
    y_pred += 1
    X_train, X_test, y_train, y_test = train_test_split(scaled_features, 
                                                        y_pred, test_size=0.1,
                                                        random_state=42)

    clf = svm.SVC(C=10, gamma=1)
    clf.fit(X_train, y_train)
    predicted_labels = clf.predict(X_test)
    print(classification_report(y_test, predicted_labels))
    facies_labels_y_test = []
    for i in range(len(np.unique(y_test))):
        facies_labels_y_test.append(str(np.unique(y_test).item(i)))
        
    conf = confusion_matrix(y_test, predicted_labels)
    display_cm(conf, facies_labels_y_test, hide_zeros=True)
    print('Facies classification accuracy: %f' % get_accuracy(conf))
    return y_pred

def compare_facies_plot_VMG(logs, arg_1, facies_colors, num_clusters):
    #make sure logs are sorted by depth
    #logs = logs.sort_values(by='Depth')
    cmap_facies = colors.ListedColormap(
            facies_colors[0:len(facies_colors[0:num_clusters])], 'indexed')
    
    ztop=logs[:,0].min(); zbot=logs[:,0].max()
    
    cluster1 = np.repeat(np.expand_dims(logs[:,6],1), 100, 1)
    #cluster2 = np.repeat(np.expand_dims(logs[arg_2].values,1), 100, 1)
    
    f, ax = plt.subplots(nrows=1, ncols=6, figsize=(9, 12))
    ax[0].plot(logs[:,1], logs[:,0], '-g')
    ax[1].plot(logs[:,2], logs[:,0], '-')
    ax[2].plot(logs[:,3], logs[:,0], '-', color='0.5')
    ax[3].plot(logs[:,4], logs[:,0], '-', color='r')
    ax[4].plot(logs[:,5], logs[:,0], '-', color='black')
    im1 = ax[5].imshow(cluster1, interpolation='none', aspect='auto',
                    cmap=cmap_facies,vmin=1,vmax=num_clusters)
    #im2 = ax[6].imshow(cluster2, interpolation='none', aspect='auto',
                    #cmap=cmap_facies,vmin=1,vmax=9)
    
    divider = make_axes_locatable(ax[5])
    cax = divider.append_axes("right", size="20%", pad=0.05)
    cbar=plt.colorbar(im1, cax=cax)
    #cbar.set_label((17*' ').join([' 1 ', ' 2 ', ' 3 ', 
                                #' 4 ', ' 5 ', ' 6 ', ' 7 ', 
                                #' 8 ', ' 9 ', ' 10 ',' 11 ',' 12 ',' 13 ',' 14 ',' 15 ']))
    #cbar.set_ticks(range(0,1)) 
    #cbar.set_ticklabels('')
    
    for i in range(len(ax)-1):
        ax[i].set_ylim(ztop,zbot)
        ax[i].invert_yaxis()
        ax[i].grid()
        ax[i].locator_params(axis='x', nbins=3)
    
    ax[0].set_xlabel("Sonic")
    ax[0].set_xlim(logs[:,1].min(),logs[:,1].max())
    ax[1].set_xlabel("density")
    ax[1].set_xlim(logs[:,2].min(),logs[:,2].max())
    ax[2].set_xlabel("GR")
    ax[2].set_xlim(logs[:,3].min(),logs[:,3].max())
    ax[3].set_xlabel("Neutron Por")
    ax[3].set_xlim(logs[:,4].min(),logs[:,4].max())
    ax[4].set_xlabel("Resistivity")
    ax[4].set_xlim(logs[:,5].min(),logs[:,5].max())
    ax[5].set_xlabel(arg_1)
    #ax[6].set_xlabel(arg_2)
    
    ax[1].set_yticklabels([]); ax[2].set_yticklabels([]); ax[3].set_yticklabels([])
    ax[4].set_yticklabels([]); ax[5].set_yticklabels([])
    ax[5].set_xticklabels([])
    #ax[6].set_xticklabels([])
    f.suptitle('B15_5-6', fontsize=14,y=0.94)
    
    plt.show()

def main():
    B15_data, feature_vectors = read_data()
    print(feature_vectors[:5,:])
    print(B15_data[:5,:])
    scaled_features = standardize(feature_vectors)
    bic_mean_cluster, der_1st, der_2nd = compute_BIC(scaled_features)
    plot_BIC(bic_mean_cluster, der_1st, der_2nd)
    '''
    num_clusters = 8                
    y_pred = validate_GMM(scaled_features, num_clusters, 
                          covariance_type='full')
    
    facies_colors = ['#FFE500', '#d2b48c','#DC7633','#6E2C00', '#FF0000', 
                     '#0000FF', '#00FFFF', '#a45dbd', '#187e03','#000000', 
                     '#dffbd7', '#fd3acd', '#f9d7f3', '#808080', '#ffffff']
    facies_labels = ['1', '2', '3', '4', '5', '6', '7','8', '9','10',
                     '11','12','13','14','15']
    facies_color_map = {}
    for ind, label in enumerate(facies_labels[0:num_clusters]):
        facies_color_map[label] = facies_colors[ind]

    blind=np.zeros((feature_vectors.shape[0],feature_vectors.shape[1]+2)*1) 
    blind[:,0] = B15_data[:,0]
    blind[:,1:4] = feature_vectors
    blind[:,4] = y_pred
    compare_facies_plot_VMG(blind, 'classification', facies_colors, 
                            num_clusters)
    '''

if __name__ == "__main__":
    main()

