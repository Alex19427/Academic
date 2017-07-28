# -*- coding: utf-8 -*-
"""
Created on Thu May 25 12:44:13 2017

@author: Alex
"""

import matplotlib.pyplot as plt
x = [3,4,9,5,9,6,4,8,5,2,5]
y = [15,20,45,25,46,32,24,36,29,5,10]
intercept= 20
slope= 10
learning_rate = 0.001

##### cost Function 

def predicted(x,y2,z):
    y = []        
    for i in x:
        y1 = z + y2 * x[i]
        y.append(y1)
    return(y)
    
#a = predicted(x,slope,intercept)

##### Sum Squared error
def square_error(x,y):
    s = []
    for i in range(len(x)):
        square = (x[i]-y[i])**2
        s.append(square)
    return(s)
#q = square_error(a,y)


##### for the first coefficient intercept 
def error_m(y,x,c2,m2):
    a = []
    for i in range(len(x)):
        a1  = -2*x[i]*(-c2-m2*x[i]+y[i])
        a.append(a1)
    return(sum(a))
w = error_m(y,x,intercept,slope)
#sum(w)

##### for the feature coefficient slope
def error_c(y,x,c,m):
    b = []
    for i in range(len(a)):
        b1 = -2*(y[i]-m*x[i]-c)
        b.append(b1)
    return(sum(b))
#r = error_c(y,x,intercept,slope)
    

m_new = []
c_new = []
q_list = []
intercept1 =[]
slope1 =[] 
for i in range(0,100):
    if i == 0:
        #intercept = intercept - m_new1`
        #slope = slope - learning_rate
        #m_new.append(m_new1)
        #c_new.append(c_new1) 
        a = predicted(x,slope,intercept)
        q = sum(square_error(a,y))
    else:
        w = error_m(y,x,intercept,slope)
        r = error_c(y,x,intercept,slope)
        c_new = intercept - w*learning_rate
        m_new = slope - r*learning_rate
        a = predicted(x,m_new,c_new)
        intercept = c_new
        slope = m_new
    q = sum(square_error(a,y))
    print("\n")
    print(intercept)
    print(slope)
    intercept1.append(m_new)
    slope1.append(c_new)
    q_list.append(q)
    #print(q_list)
    
plt.plot(q_list,"bo")
