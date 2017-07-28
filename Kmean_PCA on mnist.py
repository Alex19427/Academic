
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
from sklearn import datasets, metrics
from keras.datasets import mnist
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA


(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train.shape
#(60000, 28, 28)

data_test=x_test.reshape((len(x_test),-1))
data_train=x_train.reshape((len(x_train),-1))

data_test.shape
#(10000, 784)

kmeans = KMeans(n_clusters=10, random_state=0).fit(data_train)

kmeans
'''KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
    n_clusters=10, n_init=10, n_jobs=1, precompute_distances='auto',
    random_state=0, tol=0.0001, verbose=0)
'''
kmeans.labels_

kmeans.cluster_centers_

a=x_train[(y_train==0) & (kmeans.labels_==0)][0:20]
b=x_train[(y_train==1) & (kmeans.labels_==1)][0:20]
c=x_train[(y_train==2) & (kmeans.labels_==2)][0:20]
d=x_train[(y_train==3) & (kmeans.labels_==3)][0:20]
e=x_train[(y_train==4) & (kmeans.labels_==4)][0:20]
f=x_train[(y_train==5) & (kmeans.labels_==5)][0:20]
g=x_train[(y_train==6) & (kmeans.labels_==6)][0:20]
h=x_train[(y_train==7) & (kmeans.labels_==7)][0:20]
i=x_train[(y_train==8) & (kmeans.labels_==8)][0:20]
j=x_train[(y_train==9) & (kmeans.labels_==9)][0:20]
for char in (a,b,c,d,e,f,g,h,i,j):
    for index in range(0,len(char)):
        plt.subplot(10, 20, index + 1)
        plt.axis('off')
        plt.imshow(char[index])
    plt.show()

expected=y_train
predicted=kmeans.labels_

print("Classification report for classifier:\n%s\n"
      % (metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
print("Accuracy: %s\n" % (metrics.accuracy_score(expected, predicted)))


classifier = LogisticRegression()
classifier.fit(data_train, y_train)

expected = y_test
predicted = classifier.predict(data_test)

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
print("Accuracy: %s\n" % (metrics.accuracy_score(expected, predicted)))

pca=PCA(n_components=100)
pca.fit(data_train)

sum(pca.explained_variance_ratio_)

pca_data=pca.transform(data_train)
pca_data.shape


classifier = LogisticRegression()
classifier.fit(pca_data,y_train)

pca_test=pca.transform(data_test)
expected_pca = y_test
predicted_pca = classifier.predict(pca_test)


print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected_pca, predicted_pca)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected_pca, predicted_pca))
print("Accuracy: %s\n" % (metrics.accuracy_score(expected_pca, predicted_pca)))


































