import random
from math import fabs, sqrt, pow
import scipy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

class Knn:
    '''class predicting method'''
    def predict(
    self,
    k,
    feature_1_test,
    feature_2_test,
    feature_1_train,
    feature_2_train,
    target_test,
    target
    ):
        '''method that predicts elements in given neighbourhood'''
        def count_distances(k,arr,distances,target):
            my_max = max(arr)
            v = 0
            for m in range(len(arr)):
                if my_max==arr[m]:
                    v += 1
            if v==2 and k>1:
                k -= 1
                ass_arr = [0]*len(set(target))
                for q in range(k):
                    ass_arr[distances[q][1]] += 1
                return count_distances(k,ass_arr,distances,target)
            else:
                return arr
        appeareance_count = []
        distances = []
        for i in range(len(feature_1_test)):
            distances.append([])
            for y in range(len(feature_1_train)):
                distances[i].append([
                fabs(sqrt(pow(feature_1_train[y] - feature_1_test[i], 2)
                + pow(feature_2_train[y] - feature_2_test[i], 2))),
                target[y],
                feature_1_train[y],
                feature_2_train[y]
                ])
            distances[i].sort()
            '''choosing closests elements for testing set objects'''
            if k==1:
                appeareance_count.append(distances[i][0][1])
            else:
                arr = [0]*len(set(target))
                for q in range(k):
                    arr[distances[i][q][1]] += 1
                arr = count_distances(k,arr,distances[i],target)
                appeareance_count.append(arr)
        '''returning most frequent appearing classes for given k-neighbourhood'''
        return_predictions = []
        for i in appeareance_count:
            return_predictions.append(np.argmax(i))
        return return_predictions
    '''visualisation method which as input_coord takes [x,y] array'''
    def visualize(
    self,
    k,
    input_coord,
    feature_1_train,
    feature_2_train,
    target_test,
    target
    ):
        distances = []
        for y in range(len(feature_1_train)):
            distances.append([
            fabs(sqrt(pow(feature_1_train[y] - input_coord[0], 2)
            + pow(feature_2_train[y]-input_coord[1], 2))),
            target[y],
            feature_1_train[y],
            feature_2_train[y]
            ])
        distances.sort()
        a = len(set([i[1] for i in distances[:k]]))
        coord_nn_x = []
        koord_nn_y = []
        colors = ("c","m","y","g","b","orange","pink","olive","chocolate")
        for i in range(a):
            coord_nn_x.append([])
            koord_nn_y.append([])
        for i in distances[:k]:
            coord_nn_x[i[1]-1].append(i[2])
            koord_nn_y[i[1]-1].append(i[3])
        plt.figure(1 ,figsize=(10 ,10))
        plt.tight_layout()
        plt.xlabel('feature 1', fontsize=16)
        plt.ylabel('feature 2', fontsize=16)
        plt.title('knn', fontsize=20)
        plt.scatter(feature_1_train,feature_2_train,c="gray")
        for i in range(a):
            plt.scatter(coord_nn_x[i],koord_nn_y[i],c=colors[i])
        plt.scatter(input_coord[0],input_coord[1],c="red")
        plt.tight_layout()
        plt.show()


class Kmeans(object):

    def __init__(self):
        self = self
    '''determining centroids and clusters'''
    def fit(self,k,feature_1,feature_2,target):
        def determining_centroids(k,centr_n_clust,feature_1,feature_2,target):
            centr_n_clust[-1] = 0
            centr_n_clust[-2] = 0
            for i in range(k):
                centr_n_clust[-1] += centr_n_clust[i][1] + centr_n_clust[i][2]
                centr_n_clust[i][3] = []
            for q in range(len(feature_1)):
                distance_n_closest_centr = [(scipy.spatial.distance.euclidean(
                [
                centr_n_clust[i][1],
                centr_n_clust[i][2]],
                [feature_1[q],feature_2[q]]))
                for i in range(k)
                ]
                minimum = min(distance_n_closest_centr)
                centr_n_clust[distance_n_closest_centr.index(minimum)][3].append(
                [feature_1[q],feature_2[q],target[q]]
                )
            for i in range(len(centr_n_clust) - 2):
                x = 0
                y = 0
                for q in range(len(centr_n_clust[i][3])):
                    x += centr_n_clust[i][3][q][0]
                    y += centr_n_clust[i][3][q][1]
                centr_n_clust[i][1] = x / len(centr_n_clust[i][3])
                centr_n_clust[i][2] = y / len(centr_n_clust[i][3])
            for i in range(k):
                centr_n_clust[-2] += centr_n_clust[i][1] + centr_n_clust[i][2]
            return(centr_n_clust)
        centr_n_clust = [
        [
        i,
        random.uniform(min(feature_1),
        max(feature_1)),
        random.uniform(min(feature_2),
        max(feature_2)),[]
        ]
        for i in range(k)
        ]
        centr_n_clust.append(0)
        centr_n_clust.append(0)
        centr_n_clust=determining_centroids(
        k,
        centr_n_clust,
        feature_1,
        feature_2,target
        )
        if_stop=0
        while if_stop==0:
            if centr_n_clust[-2]==centr_n_clust[-1]:
                if_stop = 1
            else:
                centr_n_clust = determining_centroids(
                k,
                centr_n_clust,
                feature_1,
                feature_2,target
                )
        self.centr_n_clust = centr_n_clust
    '''k-means visualisation method'''
    def visualize(self):
        centr_n_clust = self.centr_n_clust
        plt.figure(1 ,figsize=(10 ,10))
        plt.tight_layout()
        plt.xlabel('feature 1', fontsize=16)
        plt.ylabel('feature 2', fontsize=16)
        plt.title('kmeans', fontsize=20)
        colors=("c","m","y","g","b","orange","pink","olive","chocolate")
        feature_1_grouped = []
        feature_2_grouped = []
        centroids_x = []
        centroids_y = []
        for i in centr_n_clust[:-2]:
            feature_1_grouped.append([])
            feature_2_grouped.append([])
            centroids_x.append(centr_n_clust[centr_n_clust.index(i)][1])
            centroids_y.append(centr_n_clust[centr_n_clust.index(i)][2])
            for q in i[3]:
                feature_1_grouped[centr_n_clust.index(i)].append(q[0])
                feature_2_grouped[centr_n_clust.index(i)].append(q[1])
        for i in range(len(centr_n_clust[:-2])):
            plt.scatter(feature_1_grouped[i],feature_2_grouped[i],c=colors[i])
        for i in range(len(centr_n_clust[:-2])):
            plt.scatter(centroids_x[i],centroids_y[i],c="r")
        plt.tight_layout()
        plt.show()
