







import numpy as np
from knn import KNN

############################################################################
# DO NOT MODIFY CODES ABOVE 
############################################################################


# TODO: implement F1 score
def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """
    assert len(real_labels) == len(predicted_labels)
    #raise NotImplementedError
    tn, fp, fn, tp = 0.0,0.0,0.0,0.0
    for i in range(0,len(real_labels)):
        real = real_labels[i]
        pred = predicted_labels[i]
        if pred==0:
            if real==0:
                tn = tn+1
            else:   #real==1
                fn = fn+1
        else:   #pred==1
            if real==1:
                tp = tp+1
            else:   #real==0
                fp = fp+1
    if tp==0:
        return 0
    else:
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        f1_score = 2*((precision*recall)/(precision+recall))
        return f1_score


class Distances:
    @staticmethod
    # TODO
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        #raise NotImplementedError
        p1 = np.array(point1)
        p2 = np.array(point2)
        return (np.sum((abs(p1 - p2)) ** 3))**(1/3)

    @staticmethod
    # TODO
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        #raise NotImplementedError
        p1 = np.array(point1)
        p2 = np.array(point2)
        return np.sqrt(np.sum((p1 - p2) ** 2))

    @staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        #raise NotImplementedError
        p1 = np.array(point1)
        p2 = np.array(point2)
        norm1 = np.sqrt(np.dot(p1, p1))
        norm2 = np.sqrt(np.dot(p2, p2))
        if norm1==0 or norm2==0:
            return 1
        else:
            return 1-(np.dot(p1, p2)/(norm1*norm2))



class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    # TODO: find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you need to try different distance functions you implemented in part 1.1 and different values of k (among 1, 3, 5, ... , 29), and find the best model with the highest f1-score on the given validation set.
		
        :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you need to try to calculate the distance. Make sure you loop over all distance functions for each k value.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] training labels to train your KNN model
        :param x_val:  List[List[int]] validation data
        :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), and model (an instance of KNN) and assign them to self.best_k,
        self.best_distance_function, and self.best_model respectively.
        NOTE: self.best_scaler will be None.

        NOTE: When there is a tie, choose the model based on the following priorities:
        First check the distance function:  euclidean > Minkowski > cosine_dist 
		(this will also be the insertion order in "distance_funcs", to make things easier).
        For the same distance function, further break tie by prioritizing a smaller k.
        """
        best_f1 = 0
        best_model = KNN(29,Distances.cosine_similarity_distance)
        best_k = 29
        best_dist_func = 'euclidean'
        for dist_name,dist_func in distance_funcs.items():
            for k in range(1, 30, 2):
                knn_model = KNN(k,dist_func)   # k, distance_function
                knn_model.train(x_train, y_train)   # features, labels
                pred = knn_model.predict(x_val) #features
                f1 = f1_score(y_val, pred)  # real_labels, predicted_labels
                if f1>best_f1:
                    best_f1 = f1
                    best_model = knn_model
                    best_k = k
                    best_dist_func = dist_name
                elif f1==best_f1:
                    dist_func_list = ['euclidean', 'minkowski', 'cosine_dist']
                    current_best = dist_func_list.index(dist_name)
                    previous_best = dist_func_list.index(best_dist_func)
                    if current_best<previous_best:
                        best_model = knn_model
                        best_k = k
                        best_dist_func = dist_name
                    elif current_best==previous_best and k<best_k:
                        best_model = knn_model
                        best_k = k
                        best_dist_func = dist_name

        # You need to assign the final values to these variables
        self.best_k = best_k
        self.best_distance_function = best_dist_func
        self.best_model = best_model
        #raise NotImplementedError

    # TODO: find parameters with the best f1 score on validation dataset, with normalized data
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is the same as "tuning_without_scaling", except that you also need to try two different scalers implemented in Part 1.3. More specifically, before passing the training and validation data to KNN model, apply the scalers in scaling_classes to both of them. 
		
        :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you need to try to calculate the distance. Make sure you loop over all distance functions for each k value.
        :param scaling_classes: dictionary of scalers (key is the scaler name, value is the scaler class) you need to try to normalize your data
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data
        :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), scaler (its name), and model (an instance of KNN), and assign them to self.best_k, self.best_distance_function, best_scaler, and self.best_model respectively.
        
        NOTE: When there is a tie, choose the model based on the following priorities:
        First check scaler, prioritizing "min_max_scale" over "normalize" (which will also be the insertion order of scaling_classes). Then follow the same rule as in "tuning_without_scaling".
        """
        
        best_f1 = 0
        best_model = KNN(29,Distances.cosine_similarity_distance)
        best_k = 29
        best_dist_func = 'euclidean'
        best_scaler = 'normalize'
        for dist_name,dist_func in distance_funcs.items():
            for scaler_name, scaler_class in scaling_classes.items():
                for k in range(1, 30, 2):
                    # scale data
                    s_class = scaler_class()
                    x_train_scaled = s_class(x_train)
                    x_val_scaled = s_class(x_val)

                    knn_model = KNN(k,dist_func)   # k, distance_function
                    knn_model.train(x_train_scaled, y_train)   # features, labels
                    pred = knn_model.predict(x_val_scaled) #features
                    f1 = f1_score(y_val, pred)  # real_labels, predicted_labels
                    if f1>best_f1:
                        best_f1 = f1
                        best_model = knn_model
                        best_k = k
                        best_dist_func = dist_name
                        best_scaler = scaler_name
                    elif f1==best_f1:
                        if scaler_name=='min_max_scale' and best_scaler=='normalize':
                            best_model = knn_model
                            best_k = k
                            best_dist_func = dist_name
                            best_scaler = scaler_name
                        elif scaler_name==best_scaler:
                            dist_func_list = ['euclidean', 'minkowski', 'cosine_dist']
                            current_best = dist_func_list.index(dist_name)
                            previous_best = dist_func_list.index(best_dist_func)
                            if current_best<previous_best:
                                best_model = knn_model
                                best_k = k
                                best_dist_func = dist_name
                                best_scaler = scaler_name
                            elif current_best==previous_best and k<best_k:
                                best_model = knn_model
                                best_k = k
                                best_dist_func = dist_name
                                best_scaler = scaler_name

        # You need to assign the final values to these variables
        self.best_k = best_k
        self.best_distance_function = best_dist_func
        self.best_scaler = best_scaler
        self.best_model = best_model
        #raise NotImplementedError


class NormalizationScaler:
    def __init__(self):
        pass

    # TODO: normalize data
    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        #raise NotImplementedError
        f = [[float(i) for i in j] for j in features]
        norm_list = []
        for each_f in f:
            p = np.array(each_f)
            norm = np.sqrt(np.dot(p, p))
            if norm>0:
                normalized_f = each_f/norm
                norm_list.append(normalized_f)
            else:
                norm_list.append(each_f)
        return norm_list


class MinMaxScaler:
    def __init__(self):
        pass

    # TODO: min-max normalize data
    def __call__(self, features):
        """
		For each feature, normalize it linearly so that its value is between 0 and 1 across all samples.
        For example, if the input features are [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]].
		This is because: take the first feature for example, which has values 2, -1, and 0 across the three samples.
		The minimum value of this feature is thus min=-1, while the maximum value is max=2.
		So the new feature value for each sample can be computed by: new_value = (old_value - min)/(max-min),
		leading to 1, 0, and 0.333333.
		If max happens to be same as min, set all new values to be zero for this feature.
		(For further reference, see https://en.wikipedia.org/wiki/Feature_scaling.)

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        #raise NotImplementedError
        f = [[float(i) for i in j] for j in features]
        norm_list = []
        min_list = []
        min_max_list = []
        len_of_f = len(f)
        for i in range(0,len(f[0])):
            sorted_f = sorted(f, key = lambda x: x[i])
            min_value = sorted_f[0][i]
            max_value = sorted_f[len_of_f-1][i]
            min_list.append(min_value)
            min_max_list.append(max_value-min_value)

        min_max_array = np.array(min_max_list, dtype=float)
        for each_f in f:
            upper = [a_i - b_i for a_i, b_i in zip(each_f, min_list)]
            upper_f = np.array(upper, dtype=float)
            norm_list.append(np.divide(upper_f, min_max_array, out=np.zeros_like(upper_f), where=min_max_array!=0))
        return norm_list
