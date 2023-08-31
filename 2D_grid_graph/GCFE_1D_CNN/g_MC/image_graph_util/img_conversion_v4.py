import numpy as np
import networkx as nx
from itertools import permutations




class image_convert:
   
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
        return np.array(temp)
    
    @staticmethod
    def to_ghersarray(lap_array):
        lap_channel=lap_array.shape[0]
        
        if lap_channel==3:
            ch1,ch2,ch3=lap_array
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
            ch1=lap_array[0]
            ch1_sum=ch1.sum(-1)[...,None]
            ch1_diag=ch1.diagonal(axis1 = 1,axis2 = 2)[...,None]
            ch1_rad=np.subtract(ch1_sum,ch1_diag)
            ch1_out=np.concatenate((ch1_rad.reshape(-1,1),ch1_diag.reshape(-1,1)),axis=1)
            chall_out=np.array([ch1_out])        

        return chall_out  
    
    @staticmethod
    def gen_gridlap(channel_input,patch_size_x,patch_size_y,ed0,ed1,adj_ary):
        ch=channel_input
        n_channels=channel_input.shape[0]
       
        temp_zero=np.zeros((n_channels,patch_size_x*patch_size_x,patch_size_y*patch_size_y))
        d_zero=np.zeros((n_channels,patch_size_x*patch_size_x,patch_size_y*patch_size_y))
        
        new_ch=ch.reshape(n_channels,patch_size_x*patch_size_y,-1)
        v1=new_ch[:,ed0]
        v2=new_ch[:,ed1]
        values=abs(v1-v2)
        revalue_s0=values.shape[0]
        revalue_s1=values.shape[1]
        re_val=values.reshape(revalue_s0,revalue_s1)
        
        temp_zero[:,ed0,ed1]=re_val
        
        d_sum=np.sum(adj_ary,axis=0)
        deg = np.diag(d_sum)
        degree_mat=d_zero+deg
        
        lap_a=abs(np.subtract(degree_mat,temp_zero))

        #print(lap_a.shape)
 
        return np.float16(lap_a)
    
    
    def to_gridlaparray(self,patch_array):
        self.patch_array=patch_array
        self.patch_length=self.patch_array.shape[0]
        self.patch_channel=self.patch_array.shape[1]
        self.patch_no=self.patch_array.shape[2]
        self.patch_size_x=self.patch_array.shape[3]
        self.patch_size_y=self.patch_array.shape[4]
        
        
        
        g = nx.generators.lattice.grid_2d_graph(self.patch_size_x,self.patch_size_y, periodic=False)
        adj_ary=nx.adjacency_matrix(g).toarray()
        edge=np.array(np.where(adj_ary==1))
        ed0=edge[0]
        ed1=edge[1]
        
        temp=[]
        for k,i in enumerate(self.patch_array):
            print(f'Converting ==> {k+1} \r', end="",flush=True)
            
            if self.patch_channel==3:
                ch1,ch2,ch3=i
                ch1_out=image_convert.gen_gridlap(ch1,self.patch_size_x,self.patch_size_y,ed0,ed1,adj_ary)
                ch2_out=image_convert.gen_gridlap(ch2,self.patch_size_x,self.patch_size_y,ed0,ed1,adj_ary)
                ch3_out=image_convert.gen_gridlap(ch3,self.patch_size_x,self.patch_size_y,ed0,ed1,adj_ary)
                
                chall_out=np.array([ch1_out,ch2_out,ch3_out])
                gherall_out=image_convert.to_ghersarray(chall_out)
            else:
                #print(i.shape)
                #print(i[0].shape)
                ch1=i[0]
                chall_out=np.array([image_convert.gen_gridlap(ch1,self.patch_size_x,self.patch_size_y,ed0,ed1,adj_ary)])
                chall_out.reshape(1,self.patch_no,(self.patch_size_x*self.patch_size_x),(self.patch_size_y*self.patch_size_y))
                gherall_out=image_convert.to_ghersarray(chall_out)
            
            temp.append(gherall_out) 
                               
        return temp
    
    
    @staticmethod
    def gen_pairwiselap(channel_input,patch_size_x,patch_size_y):
        #temp2=[]
        combo=list(permutations(range(0,patch_size_x*patch_size_y), 2))
        combo_array=np.array(combo).T
        r,c=zip(*combo)
        ed0=combo_array[0]
        ed1=combo_array[1]
        
        ch=channel_input
        n_channels=channel_input.shape[0]
       
        temp_zero=np.zeros((n_channels,patch_size_x*patch_size_x,patch_size_y*patch_size_y))
        d_zero=np.zeros((n_channels,patch_size_x*patch_size_x,patch_size_y*patch_size_y))
 
        new_ch=ch.reshape(n_channels,patch_size_x*patch_size_y,-1)
        v1=new_ch[:,ed0]
        v2=new_ch[:,ed1]
        values=abs(v1-v2)
        revalue_s0=values.shape[0]
        revalue_s1=values.shape[1]
        re_val=values.reshape(revalue_s0,revalue_s1)
        
        temp_zero[:,ed0,ed1]=re_val
        
        d_sum=np.zeros(patch_size_x*patch_size_y)+(patch_size_x*patch_size_y)
        deg = np.diag(d_sum)
        degree_mat=d_zero+deg
        
        lap_a=abs(np.subtract(degree_mat,temp_zero))

        #print(lap_a.shape)
 
        return np.float16(lap_a)
          
   
    def to_pairwiselaparray(self,patch_array):
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
                ch1_out=image_convert.gen_pairwiselap(ch1,self.patch_size_x,self.patch_size_y)
                ch2_out=image_convert.gen_pairwiselap(ch2,self.patch_size_x,self.patch_size_y)
                ch3_out=image_convert.gen_pairwiselap(ch3,self.patch_size_x,self.patch_size_y)
                
                chall_out=np.array([ch1_out,ch2_out,ch3_out])
                #chall_out=np.float16(chall_out)
                gherall_out=image_convert.to_ghersarray(chall_out)
            else:
                ch1=i[0]
                chall_out=np.array([image_convert.gen_pairwiselap(ch1,self.patch_size_x,self.patch_size_y)])
                chall_out.reshape(1,self.patch_no,(self.patch_size_x*self.patch_size_x),(self.patch_size_y*self.patch_size_y))
                gherall_out=image_convert.to_ghersarray(chall_out)
                
            temp.append(gherall_out)
                               
        return temp

      