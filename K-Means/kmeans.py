




import numpy as np

#################################
# DO NOT IMPORT OHTER LIBRARIES
#################################

def get_k_means_plus_plus_center_indices(n, n_cluster, x, generator=np.random):
    '''

    :param n: number of samples in the data
    :param n_cluster: the number of cluster centers required
    :param x: data - numpy array of points
    :param generator: random number generator. Use it in the same way as np.random.
            In grading, to obtain deterministic results, we will be using our own random number generator.


    :return: a list of length n_clusters with each entry being the *index* of a sample
             chosen as centroid.
    '''
    p = generator.randint(0, n) #this is the index of the first center
    #############################################################################
    # TODO: implement the rest of Kmeans++ initialization. To sample an example
	# according to some distribution, first generate a random number between 0 and
	# 1 using generator.rand(), then find the the smallest index n so that the 
	# cumulative probability from example 1 to example n is larger than r.
    #############################################################################
    

    # DO NOT CHANGE CODE BELOW THIS LINE
    centers = [p]

    for k in range(1,n_cluster):
        min_distance_list = []
        for j in range(0,n):
            if j not in centers:
                min_dist = np.inf
                for each_center in centers:
                    temp_dist = np.sum((x[each_center] - x[j]) ** 2)
                    if temp_dist<min_dist:
                        min_dist = temp_dist
                min_distance_list.append(min_dist)
            else:
                min_distance_list.append(0.0)

        temp_sum = sum(min_distance_list)
        norm_min_distance_list = [temp/temp_sum for temp in min_distance_list]

        r = generator.rand()
        cul_prob = 0
        current_n = 0
        while cul_prob<=r:
            if current_n not in centers:
                cul_prob = cul_prob + norm_min_distance_list[current_n]
            current_n = current_n+1
        current_n = current_n-1
        centers.append(current_n)

    # DO NOT CHANGE CODE BELOW THIS LINE
    return centers


# Vanilla initialization method for KMeans
def get_lloyd_k_means(n, n_cluster, x, generator):
    return generator.choice(n, size=n_cluster)



class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of clusters for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
    '''
    def __init__(self, n_cluster, max_iter=100, e=0.0001, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, centroid_func=get_lloyd_k_means):

        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
            returns:
                A tuple in the following order:
                  - final centroids, a n_cluster X D numpy array, 
                  - a length (N,) numpy array where cell i is the ith sample's assigned cluster's index (start from 0), 
                  - number of times you update the assignment, an Int (at most self.max_iter)
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        self.generator.seed(42)
        N, D = x.shape

        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)
        ###################################################################
        # TODO: Update means and membership until convergence 
        #   (i.e., average K-mean objective changes less than self.e)
        #   or until you have made self.max_iter updates.
        ###################################################################
        center_M = np.zeros((self.n_cluster, D))
        for i in range(0,len(self.centers)):
            center_M[i] = x[self.centers[i]]

        iter_num = 0
        obj_change = np.inf
        obj_old = np.inf
        label_idx = []
        while iter_num <= self.max_iter and obj_change >= self.e:
            # compute dist_M
            dist_M = np.zeros((self.n_cluster,N))
            for i in range(0,self.n_cluster):
                dist_row = np.sum((x-center_M[i])**2, axis = 1)
                dist_M[i] = dist_row

            record_center_M = center_M
            record_label_idx = label_idx

            # compute old_obj
            obj_new = np.sum(np.min(dist_M, axis = 0))

            # find label_idx
            label_idx = np.argmin(dist_M, axis = 0)

            # update center_M
            label_idx_dict = {}
            for i in range(0,len(label_idx)):
                idx_list = label_idx_dict.get(label_idx[i],[])
                idx_list.append(i)
                label_idx_dict[label_idx[i]]=idx_list

            for i in range(0,len(label_idx_dict)):
                idx_list = label_idx_dict.get(i,[])
                center_M[i] = np.mean(x[[idx_list]], axis = 0)

            if iter_num==0:
                obj_change = np.inf
            else:
                obj_change = abs(obj_old-obj_new)/N
            obj_old = obj_new

            iter_num = iter_num+1
        return (record_center_M, record_label_idx, iter_num)


        


class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of clusters for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator


    def fit(self, x, y, centroid_func=get_lloyd_k_means):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
            returns:
                None
            Store following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by
                    majority voting (numpy array of length n_cluster)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        self.generator.seed(42)
        N, D = x.shape
        ################################################################
        # TODO:
        # - assign means to centroids (use KMeans class you implemented, 
        #      and "fit" with the given "centroid_func" function)
        # - assign labels to centroid_labels
        ################################################################
        kmeans_model = KMeans(self.n_cluster, self.max_iter, self.e, self.generator)
        (centroids, all_cluster_idx, iter_num) = kmeans_model.fit(x, centroid_func)
        center_labels_dict = {}
        for i in range(0,N):
            center_idx = all_cluster_idx[i]
            label_freq_dict = center_labels_dict.get(center_idx,{})
            freq = label_freq_dict.get(y[i],0)
            freq = freq+1
            label_freq_dict[y[i]] = freq
            center_labels_dict[center_idx] = label_freq_dict

        centroid_labels = []
        for i in range(0,len(center_labels_dict)):
            label_freq_dict = center_labels_dict.get(i)
            centroid_labels.append(max(label_freq_dict, key=label_freq_dict.get))

        centroid_labels = np.array(centroid_labels)

        
        # DO NOT CHANGE CODE BELOW THIS LINE
        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (
            self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)

        assert self.centroids.shape == (
            self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function
            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        self.generator.seed(42)
        N, D = x.shape
        ##########################################################################
        # TODO:
        # - for each example in x, predict its label using 1-NN on the stored 
        #    dataset (self.centroids, self.centroid_labels)
        ##########################################################################
        cluster_list = []
        for i in range(0,N):
            current_x = x[i]
            dist_list = []
            for each_centroid in self.centroids:
                temp_dist = np.sum((current_x - each_centroid) ** 2)
                dist_list.append(temp_dist)

            assigned_cluster = np.argmin(dist_list)
            cluster_list.append(self.centroid_labels[assigned_cluster])
        return np.array(cluster_list)



def transform_image(image, code_vectors):
    '''
        Quantize image using the code_vectors (aka centroids)

        Return a new image by replacing each RGB value in image with the nearest code vector
          (nearest in euclidean distance sense)

        returns:
            numpy array of shape image.shape
    '''

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'
    ##############################################################################
    # TODO
    # - replace each pixel (a 3-dimensional point) by its nearest code vector
    ##############################################################################
    new_img = np.zeros(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel = image[i][j]
            dist_list = []
            for k in range(0,code_vectors.shape[0]):
                center = code_vectors[k]
                sx = np.sum(pixel**2)
                sy = np.sum(center**2)
                dist_list.append(sx -(2 * np.dot(pixel, center)) + sy)
            selected_idx = np.argmin(dist_list)
            selected_center = code_vectors[selected_idx]
            new_img[i][j] = selected_center

    return new_img
    
