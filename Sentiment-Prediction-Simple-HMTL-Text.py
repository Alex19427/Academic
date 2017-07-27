# -*- coding: utf-8 -*-
"""
Created on Tue May 16 14:02:54 2017

@author: Alex
"""
########## Import Package ##########
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from nltk.corpus import sentiwordnet as swn
from sklearn.metrics import classification_report

########## Import Data ############
o = open("E:/BDAP/Python/Python_Exam.csv","w") 
data_sent = open ("E:/BDAP/Python/sample.html","r").read()
stop = stopwords.words('english')
header = "no_words,stopwords,nouns,positive,negative,adjectives,car_name \n"
o.write(header)


def postag(s):
    w=word_tokenize(s)
    ps=nltk.pos_tag(w)
    return(ps)

def clean_html(data):
    soup = BeautifulSoup(data,'html.parser')
    soup1 = soup.find_all("title")
    soup2 = soup.find_all('body')
    soup1 = str(soup1)
    soup2 = str(soup2)
    raw = re.sub(r"(?s)<.*?>", " ", soup1)
    raw1 = re.sub(r"(?s)<.*?>", " ", soup2)
    raw2 = raw + raw1
    return raw2
data = clean_html(data_sent)
len(data)
sent = sent_tokenize(data)
len(sent)

cars = ['hyundai','jaguar','beetle','accord','civic','acura','autobacs seven',
        'daihatsu','datsun','hino','honda','infiniti','isuzu','kawasaki','lexus',
        'mazda','mitsubishi','mitsuoka','nissan','subaru','suzuki','buick',
        'cadillac','callaway','chevrolet','chrysler','detroit electric',
        'dodge','e-z-go','faraday future','fisker','ford','freightliner',
        'gem','gmc','hennessey','international harvester','jeep','karma',
        'kenworth','lincoln','local motors','navistar','oshkosh','panoz',
        'peterbilt','polaris','ram','rossion','saleen','ssc north america',
        'tesla','western star','defunct[edit]','ashok leyland','bajaj',
        'renault','peugeot','bharat benz','chinkara','dc','eicher','escorts',
        'force','hero','heavy vehicles factory','hindustan','hradyesh','kal',
        'kinetic engineering limited','lml','mahindra','maruti','ordnance factory medak',
        'premier','royal enfield','standard','swaraj','tafe','tata','tvs','9ff','alpina',
        'artega','audi','bitter','bmw','brabus','büssing','carlsson','daimler','gumpert',
        'isdera','man','mansory','mercedes-benz','multicar','neoplan','nsu','opel',
        'porsche','robur','ruf','smart','volkswagen','wiesmann','abarth','alfa romeo',
        'autobianchi','bremach','casalini','cizeta','covini engineering','de tomaso',
        'dr motor','ducati','ferrari','fiat','intermeccanica','iveco','lancia',
        'lamborghini','maserati','mazzanti','minardi','pagani','siata','vespa',
        'vignale','zagato']

'''
noun= []
car = []
for i in range(len(train_data)):
    pp=postag(train_data[i])
    for j in pp:
        if str(j[1])=='NN' or str(j[1])=='NNP' or str(j[1])=='NNS':
            noun.append(j[0])
            noun.append(train_data[i])
        if (j[0].lower() in cars):
            car.append(j[0])
            car.append(train_data[i])
print(noun)
print(car)
postag(str(noun))
'''
      
########## Creating Features ############
for i in range(len(sent)):
    pos = 0
    neu = 0
    neg = 0
    l_max=0
    polarity= []
    noun_count =0
    adj_count  =0
    car_name = 0
    word= word_tokenize(sent[i])
    wl=len(word)
    sp=[j for j in word if j in stop]
    sl=len(sp)
    pp=postag(sent[i])
    car_target=[]
    for j in pp:
        if str(j[1])=='NN' or str(j[1])=='NNP' or str(j[1])=='NNS':
            noun_count +=1
        elif str(j[1])=='JJ' or str(j[1])=='JJC':
            adj_count +=1
        if (j[0].lower() in cars):
            #car_name.append(j[0])
            car_name += 1
            if car_name > 0:
    		        car_target=1
            else:
                  car_target = 0
        if car_target == []:
            car_target = 0
        #print(car_target)
    for i1 in word:
        k = swn.senti_synsets(i1)  
        for i2 in k:
            p1 = i2.pos_score()
            n1 = i2.neg_score()
            pol=float(p1)-float(n1)
            polarity.append(float(pol))
        sort = sorted(polarity)
        if (((len(sort)>0) and (sort[0]>=0.0)) and (sum(sort)> 0)):
            l_max = max(sort)            
        elif((len(sort)>0) and (sort[0]<0.0)and(sum(sort)> 0)):
            l_max = max(sort)
        elif((len(sort)>0) and (sort[0]<0.0)and(sum(sort)< 0)):
            l_max = sort[0]
        else:
            l_max = 0    
        if l_max > 0:
            pos += 1
        elif l_max < 0:
            neg += 1
        else:
            neu +=1
    output=str(wl)+","+str(sl)+","+str(noun_count)+","+str(pos)+","+str(neg)+","+str(adj_count)+","+str(car_target)+"\n"
    o.write(str(output))
o.close()

##############  Classification ML #############

df = pd.read_csv("E:/BDAP/Python/Python_Exam.csv")
x= df[['no_words','stopwords','nouns','positive','negative','adjectives']]
y = df['car_name ']

x_train, x_cv, y_train, y_cv = train_test_split(x,y)
model=SVC()
model.fit(x_train,y_train)
predicted=model.predict(x_cv)
expected=y_cv
print (metrics.confusion_matrix(expected,predicted))
print (classification_report(expected,predicted))
