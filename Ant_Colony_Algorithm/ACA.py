# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 14:11:01 2020

@author: kocak
"""

import numpy as np
from scipy import spatial
import pandas as pd
import matplotlib.pyplot as plt
from sko.ACA import ACA_TSP

#%%
#sınıf
class Coordinate:
    def  __init__(self,x,y):
        self.x=x
        self.y=y
        
    @staticmethod
    def get_distance(a,b):
        return np.sqrt(np.abs(a.x-b.x)+np.abs(a.y-b.y))
    @staticmethod
    def get_totol_distance(coords):
        dist=0
        for first,second in zip(coords[:-1],coords[1:]):
            dist+=Coordinate.get_distance(first,second)
        dist+=Coordinate.get_distance(coords[0],coords[-1])
        return dist

#%%
#Dosya işlemleri...    
   
# Dosyayı Açma
infile = open('berlin52.tsp', 'r')


Name = infile.readline().strip().split()[1] # NAME
FileType = infile.readline().strip().split()[1] # TYPE
Comment = infile.readline().strip().split()[1] # COMMENT
Dimension = infile.readline().strip().split()[1] # DIMENSION
EdgeWeightType = infile.readline().strip().split()[1] # EDGE_WEIGHT_TYPE
infile.readline()


if __name__== '__main__':
    num_points = 52
    points_coordinate = np.zeros((num_points, 2)) #2 boyutlu dizi oluşturup içine sıfır atadım.
    #print(points_coordinate)
    coords=[]
    #print(points_coordinate)
    N = int(Dimension)
    for i in range(0, N): #sehirler olusturuluyor
        x,y = infile.readline().strip().split()[1:]
        coords.append(Coordinate(float(x),float(y)))
        for j in range(0,2):
            if(j==0):
                points_coordinate[i][j]=float(x) #içine sıfır atanan diziye koordinatlar atandı.(x)
            else:
                points_coordinate[i][j]=float(y)      
    #print(points_coordinate)  
    #print(coords)      
           
    #Şehirlerin Gösterilmesi....
    fig=plt.figure(figsize=(7,4))
    ax1=fig.add_subplot(111)
    
    for first,second in zip(coords[:-1],coords[1:]):
        ax1.plot([first.x,second.x],[first.y,second.y],'b')
    
    ax1.plot([coords[0].x,coords[-1].x],[coords[0].y,coords[-1].y])
    
    for c in coords:
        ax1.plot(c.x,c.y,"ro")
    fig.suptitle('Şehirlerin Koordinatları')
    plt.xlabel('X EKSENİ')
    plt.ylabel('Y EKSENİ')    
    plt.show()
  
# Close input file
infile.close()
#%%

ITER=500
distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')

def toplam_mesafe(routine):
    num_points, = routine.shape
    return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])


aca = ACA_TSP(func=toplam_mesafe, n_dim=num_points, 
              size_pop=350, max_iter=500,
              distance_matrix=distance_matrix)

best_x, best_y = aca.run()
fig=plt.figure(figsize=(8,5)) 
ax1=fig.add_subplot(111)
fig.suptitle('ACA Sonuç Tablosu')
best_points_ = np.concatenate([best_x, [best_x[0]]])
best_points_coordinate = points_coordinate[best_points_, :]

ax1.plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r')
fig=plt.figure(figsize=(8,5))
ax2=fig.add_subplot(111)
fig.suptitle('Mesafe Tablosu')
ax2.plot(aca.y_best_history)
for i in range(ITER):
    print (i,"Mesafe=",aca.y_best_history[i])
#pd.DataFrame(aca.y_best_history).cummin().plot(ax=ax2)
plt.show()










