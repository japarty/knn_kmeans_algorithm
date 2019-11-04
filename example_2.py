from knn_kmeans_algorithm import Knn,Kmeans
from sklearn.datasets import load_iris,load_breast_cancer
from sklearn.model_selection import train_test_split

'''data load'''
breast_data = load_breast_cancer()
data = breast_data['data']
target = breast_data['target']
data,data_test,target,target_test = train_test_split(data,target,random_state=42)
'''data split'''
feature_1_train = [i[0] for i in data]
feature_2_train = [i[3] for i in data]
feature_1_test = [i[0] for i in data_test]
feature_2_test = [i[3] for i in data_test]
feature_1 = [i[0] for i in data]
feature_2 = [i[3] for i in data]


model = Knn()
a=model.predict(6,feature_1_test,feature_2_test,feature_1_train,feature_2_train,target_test,target)
model.visualize(9,[6,3.5],feature_1_train,feature_2_train,target_test,target)
model2 = Kmeans()
model2.fit(3,feature_1,feature_2,target)
model2.visualize()
