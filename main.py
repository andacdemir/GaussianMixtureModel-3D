import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.preprocessing import StandardScaler
from sklearn import mixture
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from classification_utilities import display_cm, display_adj_cm
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
import plotly.plotly as py
import plotly.graph_objs as go

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
def compute_BIC(scaled_features, covariance_type='full', max_clusters=12):
    cluster_ref = np.arange(1, max_clusters+1)
    iters = 10 # number of iterations

    bic_mean_cluster = np.zeros((len(cluster_ref),)*1) 
    bic_std_cluster = np.zeros((len(cluster_ref),)*1) 

    for i in tqdm(range(max_clusters)):  
        bic = []
        for _ in range(iters):
            gmm = mixture.GaussianMixture(cluster_ref[i-1], covariance_type)\
                                                       .fit(scaled_features)
            bic.append(gmm.bic(scaled_features))
        bic_mean_cluster[i-1] = np.mean(bic)
        bic_std_cluster[i-1] = np.std(bic)

    der_1st, der_2nd = derivatives_calc(bic_mean_cluster)
    return bic_mean_cluster, der_1st, der_2nd   

'''
    Plots the BIC values corresponding to each cluster
    Additionally plots the first and second derivatives of the BIC curve 
'''
def plot_BIC(bic_mean_cluster, der_1st, der_2nd, max_clusters=10):    
    plt.figure(figsize=(20,10))

    plt.subplot(321, facecolor='darkslategray')
    plt.plot(bic_mean_cluster, 'C1', marker='o')
    plt.xlim(1, max_clusters)
    plt.ylabel("BIC", color='C1')

    plt.subplot(323, facecolor='darkslategray')
    plt.plot(der_1st, 'C1', marker='o')
    plt.xlim(1, max_clusters)
    plt.ylabel("1st derivative", color='C1')

    plt.subplot(325, facecolor='darkslategray')
    plt.plot(der_2nd, 'C1', marker='o')
    plt.xlim(1, max_clusters)
    plt.ylabel("2nd derivative", color='C1')

    plt.savefig('BIC_firstder_secondder.png')
    plt.show()

'''
    Confusion matrix should be a diagonal matrix.
    Computes the accuracy by dividing the sum of diagonal elements of the 
    confusion matrix to the sum of all elements.
'''
def get_accuracy(confusion_matrix):
    total_correct = 0.
    num_classes = confusion_matrix.shape[0]
    for i in range(num_classes):
        total_correct += confusion_matrix[i][i]
    acc = total_correct/sum(sum(confusion_matrix))
    
    return acc

def validate_GMM(B15_data, scaled_features, num_clusters, 
                 covariance_type='full'):
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
    # return a dictionary with keys being the log names and 
    # facies classificaton predictions being the values. 
    d = dict()
    for key, value in zip(B15_data[:,-1], y_pred):
        if key in d.keys():
            d[key].append(value)
        else:
            d[key] = []
            d[key].append(value)
    return d

'''
    Gets the dictionary d and returns all the facies in the entire
    dataset in an order as a np.array:
'''
def get_facies(d):
    facies = [i for j in list(d.values()) for i in j]
    return facies

def compare_facies_plot_VMG(logs, arg_1, facies_colors, num_clusters, labels): 
    # make sure logs are sorted by depth
    # logs = logs.sort_values(by='Depth')
    cmap_facies = colors.ListedColormap(
            facies_colors[0:len(facies_colors[0:num_clusters])], 'indexed')
    
    ztop = logs[:,0].min(); zbot=logs[:,0].max()
    cluster1 = np.expand_dims(logs[:,4],1)
    
    f, ax = plt.subplots(nrows=1, ncols=4, figsize=(9, 12))
    ax[0].plot(logs[:,1], logs[:,0], color='r')
    ax[1].plot(logs[:,2], logs[:,0], color='g')
    ax[2].plot(logs[:,3], logs[:,0], color='b')
   
    im1 = ax[3].imshow(cluster1, interpolation='none', aspect='auto',
                       cmap=cmap_facies, vmin=1, vmax=num_clusters)     
    divider = make_axes_locatable(ax[3])
    cax = divider.append_axes("right", size="20%", pad=0.05)
    cbar = plt.colorbar(im1, cax=cax, 
                        ticks=[x+1 for x in list(range(num_clusters))])
        
    # Prints the log id, well depth and correcsponding classification
    # of the formation:
    for i in range(len(logs[:,-1])):
        if i == 0:
            print("Welllog ID: %s" %labels[-1])
        print("Depth: %.3f Facies Classification %d" %(logs[i,0], logs[i,4]))

    for i in range(len(ax)-1):
        ax[i].set_ylim(ztop,zbot)
        ax[i].invert_yaxis()
        ax[i].grid()
        ax[i].locator_params(axis='x', nbins=3)
    
    ax[0].set_xlabel("Gamma Ray")
    ax[0].set_xlim(logs[:,1].min(), logs[:,1].max())
    ax[1].set_xlabel("Neutron Porosity")
    ax[1].set_xlim(logs[:,2].min(), logs[:,2].max())
    ax[2].set_xlabel("Bulk Density")
    ax[2].set_xlim(logs[:,3].min(), logs[:,3].max())
    ax[3].set_xlabel(arg_1)
    
    ax[1].set_yticklabels([]); ax[2].set_yticklabels([]) 
    ax[3].set_yticklabels([]); ax[3].set_xticklabels([])
    textstr = '\n'.join((r'WELL ID: %s' % (labels[-1],),
                         r'$\mathrm{Latitude}: %.8f$' %(labels[0],),
                         r'$\mathrm{Longitude}: %.8f$' %(labels[1],)))
    f.suptitle(textstr)   
    plt.savefig('results\%s.pdf'%(labels[-1]))
    plt.show()

'''
    Plots the distribution of the training data by facies.
    Takes d as an argument which is a dictionary with keys being the 
    lognames and values being the facies classified at different depths.
'''
def plt_facies_distribution(facies, num_clusters):
    # counts the number of occurrences of facies as a value in the dictionary
    # appends them in a list
    classes = list(range(1, num_clusters+1))
    occurs = []
    for i in range(num_clusters):
        occurs.append(facies.count(i+1))

    df = pd.DataFrame({'Facies': classes,'Occurrence': occurs})
    ax = df.plot.bar(x='Facies', y='Occurrence', 
                     title='Distribution of Training Data by Facies')

def plt_3D(B15_data, facies, num_clusters, truncate=True):
    columns = ['Latitude', 'Longitude', 'Depth', 'Cluster']
    n_samples = len(facies)
    depths = np.asarray(B15_data[:,0]).reshape((n_samples,1))
    facies = np.asarray(facies).reshape((n_samples,1))
    df = np.concatenate((depths, B15_data[:,4:6], facies), axis=1)
    df = pd.DataFrame(data=df, columns=columns)
    if truncate==True:
        df = df.sample(frac=1) # shuffles the df randomly
        df = df.head(35000)    # slices first 35000 samples of the df

    data = []
    clusters = []
    # Add more colors if num_clusters > 5 !!
    colors = ['rgb(228,26,28)','rgb(55,126,184)','rgb(77,175,74)', 
              'rgb(233,247,32)', 'rgb(0,0,0)']

    for i in range(num_clusters):
        name = df['Cluster'].unique()[i]
        color = colors[i]
        x = df[df['Cluster'] == name]['Latitude']
        y = df[df['Cluster'] == name]['Longitude']
        z = df[df['Cluster'] == name]['Depth']
        
        trace = dict(
            name = name,
            x=x, y=y, z=z,
            type="scatter3d",    
            mode='markers',
            marker=dict(size=3, color=color, line=dict(width=0)))
        data.append(trace)

    layout = dict(
        width=800,
        height=550,
        autosize=False,
        title='Iris dataset',
        scene=dict(
            xaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            yaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            zaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            aspectratio=dict(x=1,y=1,z=1),
            aspectmode='manual'        
        ),
    )

    fig = dict(data=data, layout=layout)
    url = py.plot(fig, filename='Facies Classification 3D', validate=False)

'''
    scaled_features is a np.array with:
    columns = [Gamma_Ray, Neutron_Porosity, Bulk_Density]
    Plot:
    GR vs Porosity
    GR vs Density
    Porosity vs Density
    where facies are a cluster in the plot
    There are more than 350K samples in the data, 
    if truncate=True, only plots for 500 samples from the shuffled
    dataframe.
'''
def plt_cross_correlation(scaled_features, facies, num_clusters, 
                          truncate=True):
    columns = ['Gamma Ray', 'Porosity', 'Density', 'FACIES']
    facies = np.asarray(facies).reshape((len(facies),1))
    data = np.concatenate((scaled_features, facies), axis=1) 
    data = pd.DataFrame(data=data, columns=columns)

    if truncate==True:
        data = data.sample(frac=1) # shuffles the df randomly
        data = data.head(500)      # slices first 500 samples of the df

    # sets font size of labels on matplotlib plots
    plt.rc('font', size=16)
    # sets style of plots
    sns.set_style('white')
    # defines a custom palette
    customPalette = ['#630C3A', '#39C8C6', '#D3500C', '#FFB139', '#0c6335']
    sns.set_palette(customPalette)
    sns.palplot(customPalette)
    
    # plots data with seaborn 
    facet = sns.lmplot(data=data, x='Porosity', y='Gamma Ray', hue='FACIES', 
                       fit_reg=False, legend=True, legend_out=True)
    facet = sns.lmplot(data=data, x='Porosity', y='Density', hue='FACIES', 
                       fit_reg=False, legend=True, legend_out=True)
    facet = sns.lmplot(data=data, x='Density', y='Gamma Ray', hue='FACIES', 
                       fit_reg=False, legend=True, legend_out=True)

def main():
    B15_data, feature_vectors = read_data()
    scaled_features = standardize(feature_vectors)
    #bic_mean_cluster, der_1st, der_2nd = compute_BIC(scaled_features)
    #plot_BIC(bic_mean_cluster, der_1st, der_2nd)
    num_clusters = 5  
    d = validate_GMM(B15_data, scaled_features, num_clusters, 
                     covariance_type='full')
    facies = get_facies(d)
    #plt_facies_distribution(facies, num_clusters)
    #plt_cross_correlation(scaled_features, facies, num_clusters)
    plt_3D(B15_data, facies, num_clusters)

    facies_colors = ['#FFE500', '#d2b48c','#DC7633','#6E2C00', '#FF0000', 
                     '#0000FF', '#00FFFF', '#a45dbd', '#187e03','#000000', 
                     '#dffbd7', '#fd3acd', '#f9d7f3', '#808080', '#ffffff']
    facies_labels = ['1', '2', '3', '4', '5', '6', '7','8', '9','10',
                     '11', '12', '13', '14', '15']
    facies_color_map = {}
    for ind, label in enumerate(facies_labels[0:num_clusters]):
        facies_color_map[label] = facies_colors[ind]

    for key in d:
        indices = [i for i, x in enumerate(B15_data[:,-1]) if x == key]      
        start, end = indices[0], indices[-1]
        n_rows = end - start + 1
        blind = np.zeros((n_rows, feature_vectors.shape[1]+2)*1) 
        blind[:,0] = B15_data[start:end+1,0]
        blind[:,1:4] = feature_vectors[start:end+1,:]
        blind[:,4] = d[key]
        lat, lon = B15_data[start, 4], B15_data[start, 5]
        name = B15_data[start, -1]
        labels = [lat, lon, name]
        compare_facies_plot_VMG(blind, 'classification', facies_colors, 
                                num_clusters, labels)
  

if __name__ == "__main__":
    main()

