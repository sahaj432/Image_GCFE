#new

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import torch
import networkx as nx




class image:
   
    def to_patcharray(self,x_in,patch_size=(2,2)):
        self.x_in=x_in
        self.patch_size=patch_size
        temp=[]
        for k,i in enumerate(self.x_in):
            print(f'Converting ==> {k+1} \r', end="",flush=True)
            temp1=[]
            for j in range(i.shape[2]):
                tmp = np.array(np.split(i[:,:,j], i.shape[0]/self.patch_size[1], axis=1))
                div_final = np.vstack(np.split(tmp, tmp.shape[1]/self.patch_size[0], axis=1))
                temp1.append(div_final)
            temp1=np.array(temp1)
            temp.append(temp1)
        return temp
    
    @staticmethod
    def gen_lap(channel_input,patch_size_x,patch_size_y):
        temp2=[]
        #print(patch_array.shape)
        for k in channel_input:
            #print(k.shape)
            temp_zero=np.zeros((patch_size_x*patch_size_x,patch_size_y*patch_size_y))
            g = nx.generators.lattice.grid_2d_graph(patch_size_x,patch_size_y, periodic=False)
            adj_ary=nx.adjacency_matrix(g).toarray()
            edge=np.array(np.where(adj_ary==1))
            ed0=edge[0]
            ed1=edge[1]
            loc=list(zip(ed0,ed1))
            r,c=zip(*loc)
            flat=k.flatten()
            
            d_zero=np.zeros((patch_size_x*patch_size_x,patch_size_y*patch_size_y))
            values=abs(flat[ed0]-flat[ed1])
        
            temp_zero[r,c]=values
            d_sum=np.sum(adj_ary,axis=1)
            np.fill_diagonal(d_zero,d_sum)
            
            lap_a=d_zero-temp_zero
            
            temp2.append(lap_a)
        return temp2
        
   
    
    def to_laparray(self,patch_array):
        self.patch_array=patch_array
        self.patch_length=self.patch_array.shape[0]
        self.patch_channel=self.patch_array.shape[1]
        self.patch_no=self.patch_array.shape[2]
        self.patch_size_x=self.patch_array.shape[3]
        self.patch_size_y=self.patch_array.shape[4]
        
        temp=[]
       
        
        for k,i in enumerate(self.patch_array):
            print(f'Converting ==> {k+1} \r', end="",flush=True)
            
            if self.patch_channel==3:
                ch1,ch2,ch3=i
                ch1_out=image.gen_lap(ch1,self.patch_size_x,self.patch_size_y)
                ch2_out=image.gen_lap(ch2,self.patch_size_x,self.patch_size_y)
                ch3_out=image.gen_lap(ch3,self.patch_size_x,self.patch_size_y)
                
                chall_out=np.array([ch1_out,ch2_out,ch3_out])
                chall_out=np.float16(chall_out)
            else:
                ch1=i[0]
                chall_out=np.array([image.gen_lap(ch1,self.patch_size_x,self.patch_size_y)])
                chall_out.reshape(1,self.patch_no,(self.patch_size_x*self.patch_size_x),(self.patch_size_y*self.patch_size_y))
                chall_out=np.float16(chall_out)
            
            temp.append(chall_out)
           
           
                               
        return temp
    
    
    def to_ghersarray(self,lap_array):
        self.lap_array=lap_array
        self.lap_length=self.lap_array.shape[0]
        self.lap_channel=self.lap_array.shape[1]
        self.lap_no=self.lap_array.shape[2]
        self.lap_size_x=self.lap_array.shape[3]
        self.lap_size_y=self.lap_array.shape[4]
        
        temp=[]
       
        for k, i in enumerate(self.lap_array):
            #print(i.shape)
            print(f'Converting ==> {k+1} \r', end="",flush=True)
            if self.lap_channel==3:
                ch1,ch2,ch3=i
                ch1_sum=ch1.sum(-1)[...,None]
                ch2_sum=ch2.sum(-1)[...,None]
                ch3_sum=ch3.sum(-1)[...,None]
                ch1_diag=ch1.diagonal(axis1 = 1,axis2 = 2)[...,None]
                ch2_diag=ch2.diagonal(axis1 = 1,axis2 = 2)[...,None]
                ch3_diag=ch3.diagonal(axis1 = 1,axis2 = 2)[...,None]
                ch1_rad=np.subtract(ch1_sum,ch1_diag)
                ch2_rad=np.subtract(ch2_sum,ch2_diag)
                ch3_rad=np.subtract(ch3_sum,ch3_diag)
                ch1_out=np.concatenate((ch1_rad.reshape(-1,1),ch1_diag.reshape(-1,1)),axis=1)
                ch2_out=np.concatenate((ch2_rad.reshape(-1,1),ch2_diag.reshape(-1,1)),axis=1)
                ch3_out=np.concatenate((ch3_rad.reshape(-1,1),ch3_diag.reshape(-1,1)),axis=1)
                chall_out=np.array([ch1_out,ch2_out,ch3_out])
            else:
                ch1=i[0]
                ch1_sum=ch1.sum(-1)[...,None]
                ch1_diag=ch1.diagonal(axis1 = 1,axis2 = 2)[...,None]
                ch1_rad=np.subtract(ch1_sum,ch1_diag)
                ch1_rad=np.subtract(ch1_sum,ch1_diag)
                ch1_out=np.concatenate((ch1_rad.reshape(-1,1),ch1_diag.reshape(-1,1)),axis=1)
                chall_out=np.array([ch1_out])

            temp.append(chall_out)


        return temp


            
            

        
  