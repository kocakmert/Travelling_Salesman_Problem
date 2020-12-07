# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 15:33:29 2020

@author: kocak
"""
import random
import numpy as np
import matplotlib.pyplot as plt

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

class PSO(object):
    def __init__(self, num_city, data):
        self.iter_max = 2000 
        self.num =70
        self.num_city = num_city 
        self.location = data 

        self.dis_mat = self.compute_dis_mat(num_city, self.location)  
     
        self.particals = self.random_init(self.num, num_city)
        self.lenths = self.compute_paths(self.particals)
        
        init_l = min(self.lenths)
        init_index = self.lenths.index(init_l)
        init_path = self.particals[init_index]
        
        self.local_best = self.particals
        self.local_best_len = self.lenths
        
        self.global_best = init_path
        self.global_best_len = init_l
        
        self.best_maliyet = self.global_best_len
        self.best_path = self.global_best
        
        self.iter_x = [0]
        self.iter_y = [init_l]

    def random_init(self, num_total, num_city):
        tmp = [x for x in range(num_city)]
        result = []
        for i in range(num_total):
            random.shuffle(tmp)
            result.append(tmp.copy())
        return result
    def compute_dis_mat(self, num_city, location):
        dis_mat = np.zeros((num_city, num_city))
        for i in range(num_city):
            for j in range(num_city):
                if i == j:
                    dis_mat[i][j] = np.inf
                    continue
                a = location[i]
                b = location[j]
                tmp = np.sqrt(sum([(x[0] - x[1]) ** 2 for x in zip(a, b)]))
                dis_mat[i][j] = tmp
        return dis_mat
    
    def compute_pathlen(self, path, dis_mat):
        a = path[0]
        b = path[-1]
        result = dis_mat[a][b]
        for i in range(len(path) - 1):
            a = path[i]
            b = path[i + 1]
            result += dis_mat[a][b]
        return result
    
    def compute_paths(self, paths):
        result = []
        for one in paths:
            length = self.compute_pathlen(one, self.dis_mat)
            result.append(length)
        return result

    def sürü_degerlendirme(self):
        min_lenth = min(self.lenths)
        min_index = self.lenths.index(min_lenth)
        cur_path = self.particals[min_index]

        if min_lenth < self.global_best_len:
            self.global_best_len = min_lenth
            self.global_best = cur_path

        for i, l in enumerate(self.lenths):
            if l < self.local_best_len[i]:
                self.local_best_len[i] = l
                self.local_best[i] = self.particals[i]

    def cross(self, cur, best):
        one = cur.copy()
        l = [t for t in range(self.num_city)]
        t = np.random.choice(l,2)
        x = min(t)
        y = max(t)
        cross_part = best[x:y]
        tmp = []
        for t in one:
            if t in cross_part:
                continue
            tmp.append(t)

        one = tmp + cross_part
        l1 = self.compute_pathlen(one, self.dis_mat)
        one2 = cross_part + tmp
        l2 = self.compute_pathlen(one2, self.dis_mat)
        if l1<l2:
            return one, l1
        else:
            return one, l2

    def mutate(self, one):
        one = one.copy()
        l = [t for t in range(self.num_city)]
        t = np.random.choice(l, 2)
        x, y = min(t), max(t)
        one[x], one[y] = one[y], one[x]
        l2 = self.compute_pathlen(one,self.dis_mat)
        return one, l2


    def pso(self):
        for sayac in range(1, self.iter_max):

            for i, one in enumerate(self.particals):
                tmp_l = self.lenths[i]

                new_one, new_l = self.cross(one, self.local_best[i])
                if new_l < self.best_maliyet:
                    self.best_maliyet = tmp_l
                    self.best_path = one

                if new_l < tmp_l or np.random.rand()<0.1:
                    one = new_one
                    tmp_l = new_l

                new_one, new_l = self.cross(one, self.global_best)

                if new_l < self.best_maliyet:
                    self.best_maliyet = tmp_l
                    self.best_path = one
                if new_l < tmp_l or np.random.rand()<0.1:
                    one = new_one
                    tmp_l = new_l
                one, tmp_l = self.mutate(one)

                if new_l < self.best_maliyet:
                    self.best_maliyet = tmp_l
                    self.best_path = one

                if new_l < tmp_l or np.random.rand()<0.1:
                    one = new_one
                    tmp_l = new_l

                self.particals[i] = one
                self.lenths[i] = tmp_l

            self.sürü_degerlendirme()

            if self.global_best_len < self.best_maliyet:
                self.best_maliyet = self.global_best_len
                self.best_path = self.global_best
            print(sayac,"Mesafe:",self.best_maliyet)
            self.iter_x.append(sayac)
            self.iter_y.append(self.best_maliyet)

        return self.best_maliyet, self.best_path

    def run(self):
        best_length, best_path = self.pso()
        fig=plt.figure(figsize=(8,5))
        ax4=fig.add_subplot(111)
        fig.suptitle('Mesafe Tablosu')
        ax4.plot(self.iter_x, self.iter_y)
        plt.plot()
        return self.location[best_path], best_length

def read_tsp(path):
    infile = open(path, 'r')
    coords = np.zeros((52, 2)) #2 boyutlu dizi oluşturup içine sıfır atadım.
    Name = infile.readline().strip().split()[1] # NAME
    FileType = infile.readline().strip().split()[1] # TYPE
    Comment = infile.readline().strip().split()[1] # COMMENT
    Dimension = infile.readline().strip().split()[1] # DIMENSION
    EdgeWeightType = infile.readline().strip().split()[1] # EDGE_WEIGHT_TYPE
    infile.readline()
    
    N = int(Dimension)
    for i in range(0, N): #sehirler olusturuluyor
        x,y = infile.readline().strip().split()[1:]
        koor.append(Coordinate(float(x),float(y)))
        for j in range(0,2):
            if(j==0):
               coords[i][j]=float(x) #içine sıfır atanan diziye koordinatlar atandı.(x)
            else:
                coords[i][j]=float(y)  
    infile.close()
    data = coords
    return data
#%%
if __name__== '__main__':
    
    koor=[]
    data = read_tsp('berlin52.tsp')
#Şehirlerin Gösterilmesi....
    fig=plt.figure(figsize=(7,4))
    ax1=fig.add_subplot(111)
    
    for first,second in zip(koor[:-1],koor[1:]):
        ax1.plot([first.x,second.x],[first.y,second.y],'b')
    
    ax1.plot([koor[0].x,koor[-1].x],[koor[0].y,koor[-1].y])
    
    for c in koor:
        ax1.plot(c.x,c.y,"ro")
    fig.suptitle('Şehirlerin Koordinatları')
    plt.xlabel('X EKSENİ')
    plt.ylabel('Y EKSENİ')    
    plt.show()
    
#%%
pso = PSO(num_city=data.shape[0], data=data.copy())
Best_path, Best = pso.run()
print(Best)
fig=plt.figure(figsize=(7,4))
ax1=fig.add_subplot(111)
fig.suptitle('PSO Sonuç')
Best_path = np.vstack([Best_path, Best_path[0]])
ax1.plot(Best_path[:, 0], Best_path[:, 1],
           marker='o', markerfacecolor='b', color='c', linestyle='-')

#plt.title('Sonuç')
plt.show()
#%%





































