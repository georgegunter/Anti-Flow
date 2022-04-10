from sklearn.cluster import KMeans


def k_means_cluster(max_losses,cluster_diff=0.1):

    min_l = np.min(max_losses)
    max_l = np.max(max_losses)
    normalize_losses = (max_losses-min_l)/(max_l-min_l)
    X = normalize_losses
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X.reshape(-1,1))
    labels = kmeans.labels_
    cluster_centroids = kmeans.cluster_centers_

    positive_labels = []
    negative_labels = []

    for i in range(len(X)):
        l = X[i]
        label = labels[i]
        if(label==0):negative_labels.append(l)
        else:positive_labels.append(l) 

    if(np.min(positive_labels)-cluster_diff > np.max(negative_labels)):
        return labels,cluster_centroids
    else:
        return np.zeros_like(labels)


def 