# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 14:37:17 2020

@author: kocak
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial
from sko.SA import SA_TSP
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

distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')
print(distance_matrix)
distance_matrix = distance_matrix * 111000  # 1 derece of lat/lon ~ = 111000m
print(distance_matrix)

def toplam_mesafe(routine):
    num_points, = routine.shape
    return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])


# %%


sa_tsp = SA_TSP(func=toplam_mesafe, x0=range(num_points), T_max=300, T_min=1, L=10 * num_points,max_stay_counter=2000)

best_points, best_distance = sa_tsp.run()
print(best_points, best_distance, toplam_mesafe(best_points))

fig=plt.figure(figsize=(8,5))
ax1=fig.add_subplot(111)
fig.suptitle('Mesafe Tablosu')

best_points_ = np.concatenate([best_points, [best_points[0]]])
best_points_coordinate = points_coordinate[best_points_, :]
ax1.plot(sa_tsp.best_y_history)
ax1.set_xlabel("Iteration")
ax1.set_ylabel("Mesafe")
for i in range(2000):
    print (i,"Mesafe=",sa_tsp.best_y_history[i] /111000)

fig=plt.figure(figsize=(8,5))
fig.suptitle('SA Sonuç')
ax2=fig.add_subplot(111)
ax2.plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1],
           marker='o', markerfacecolor='b', color='c', linestyle='-')

ax2.set_xlabel("Longitude")
ax2.set_ylabel("Latitude")
plt.show()




 
     