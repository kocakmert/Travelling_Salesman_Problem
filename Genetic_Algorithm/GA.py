# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 22:22:59 2020

@author: kocak
"""
##Kütüphane importları

import numpy as np
import matplotlib.pyplot as plt
from sko.GA import GA_TSP
from scipy import spatial


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
num_points = 52
ITER=2000

distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')

def toplam_mesafe(routine):
    num_points, = routine.shape
    return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])

   
Genetic_Algo = GA_TSP(func=toplam_mesafe, n_dim=num_points, size_pop=100, max_iter=2000, prob_mut=220)

best_points, best_distance = Genetic_Algo.run()

fig=plt.figure(figsize=(8,5))
ax1=fig.add_subplot(111)
fig.suptitle('GA Sonuç Tablosu')
en_iyi = np.concatenate([best_points, [best_points[0]]])
en_iyi_noktakoordinat = points_coordinate[en_iyi, :]
print(en_iyi_noktakoordinat)
ax1.plot(en_iyi_noktakoordinat[:, 0], en_iyi_noktakoordinat[:, 1], 'o-r')
fig=plt.figure(figsize=(8,5))
ax2=fig.add_subplot(111)
fig.suptitle('Maliyet Tablosu')
ax2.plot(Genetic_Algo.generation_best_Y)
for i in range(ITER):
    print (i,"Mesafe=",Genetic_Algo.generation_best_Y[i])
plt.show()


















