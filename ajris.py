from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from math import fabs, sqrt, pow
import numpy as np
'''wybór największej wartości dla k>1'''
def licz(k,tab,length):
    maks=max(tab)
    v=0
    for m in range(len(tab)):
        if maks==tab[m]:
            v+=1
        if v==2 and k>1:
            k-=1
            for q in range(k-1):
                tab[length[q][1]]+=1
            return licz(k,tab,length)
        else:
            return tab


'''data load'''
iris_data= load_iris()
data = iris_data['data']
target = iris_data['target']
X_train,X_test,y_train,y_test = train_test_split(data,target,random_state=42)

k=int(input("Ile elementów ma być brane pod uwagę?:"))

'''podzial na listy'''
trainx=[i[0] for i in X_train]
trainy=[i[1] for i in X_train]
testx=[i[0] for i in X_test]
testy=[i[1] for i in X_test]

'''wyliczanie odległości i sort'''
def bla(k=k,trainx=trainx,trainy=trainy,testx=testx,testy=testy,y_test=y_test):
    what=[]
    length=[]
    for i in range(len(testx)):
        length.append([])
        for y in range(len(trainx)):
            length[i].append([fabs(sqrt(pow(trainx[y]-testx[i],2)+pow(trainy[y]-testy[i],2))),y_train[y]])
        length[i].sort()
        '''wybór największej wartości'''
        tab=[0]*k
        if k==1:
            what.append(length[i][0][1])
        else:
            for q in range(k):
                tab[length[i][q][1]]+=1
            tab=licz(k,tab,length[i])
            what.append(tab)

    p=0
    for i in range(len(y_test)):
        a=np.argmax(what[i])
        if a==y_test[i]:
            p+=1
    return round(p/len(y_test),2)

print(bla())
