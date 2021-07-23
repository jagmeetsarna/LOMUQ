from argparse import ArgumentParser
import h5py
import pandas as pd
import os
import random   
import matplotlib.pyplot as plt
import math
import numpy as np

os.chdir(r"D:\TOPIOS Data\data\twentyone") #Please change the directory to where your particle dataset is


#--------------------------------------------------------
density_data = h5py.File("particlecount.h5", "r")
width, height = np.shape(density_data['pcount'][0])
particle_data = h5py.File("particles.h5", "r")


px = particle_data['p_x'][()]
py = particle_data['p_y'][()]
px_attrs = particle_data['p_x']
py_attrs = particle_data['p_y'].attrs.keys()


class BoundingBox(object):
    def __init__(self, *args, **kwargs):
        self.lat_min = None
        self.lon_min = None
        self.lat_max = None
        self.lon_max = None


def get_bounding_box(latitude_in_degrees, longitude_in_degrees, half_side_in_miles):
    assert half_side_in_miles > 0
    assert latitude_in_degrees >= -90.0 and latitude_in_degrees  <= 90.0
    assert longitude_in_degrees >= -180.0 and longitude_in_degrees <= 180.0

    half_side_in_km = half_side_in_miles * 1.609344
    lat = math.radians(latitude_in_degrees)
    lon = math.radians(longitude_in_degrees)

    radius  = 6371
    # Radius of the parallel at given latitude
    parallel_radius = radius*math.cos(lat)

    lat_min = lat - half_side_in_km/radius
    lat_max = lat + half_side_in_km/radius
    lon_min = lon - half_side_in_km/parallel_radius
    lon_max = lon + half_side_in_km/parallel_radius
    rad2deg = math.degrees

    box = BoundingBox()
    box.lat_min = rad2deg(lat_min)
    box.lon_min = rad2deg(lon_min)
    box.lat_max = rad2deg(lat_max)
    box.lon_max = rad2deg(lon_max)

    return (box)



#initializing    
#Reformat long-lat in x-y format for image

class BoundedParticleCount:
    
    def __init__(self,lat,long,bbox,sample_size):
        self.lat = lat
        self.long = long
        self.bbox = bbox
        self.sample_size = sample_size
        
    
    def boundedBoxRandomParticles(self):

        #extends = (float(px_attrs['min']), float(px_attrs['max']), float(py_attrs['min']), float(py_attrs['max']))
        bound_box = get_bounding_box(self.lat,self.long,self.bbox)
        #bound_box = get_bounding_box(5,64,500)
        inside_boundingbox=[]
        inside_boundingbox_random = []
        randomList=[]
        #Timesteps
        
        timeframe = len(density_data['pcount'])-1
        
        arr = []
       
        for i in range(len(px)):
        
            if bound_box.lat_min <= px[i,0] and px[i,0] <= bound_box.lat_max and bound_box.lon_min <= py[i,0] and py[i,0] <= bound_box.lon_max :
                # print(px[i,0])
                inside_boundingbox.append([px[i,0],py[i,0]])
                arr.append(i)
        
                
        randomList.append(random.sample(arr,round((len(inside_boundingbox)-1)*self.sample_size))) #random sampling
       
        for k in range(timeframe):
            
            for i in randomList[0]:
                #print(k)
                inside_boundingbox_random.append([px[i,k],py[i,k],k])
                        
        return inside_boundingbox_random,randomList
    
    def convertParticlesToParticlesCount(self,inside_boundingbox_random):
        
        
        hf = h5py.File('data_topios.h5','w')
       
        days_dict = {}
        
        for r in inside_boundingbox_random:
            try:
                
                days_dict[r[2]].append([r[0],r[1]]) 
            except KeyError:
                 days_dict[r[2]] = [[r[0],r[1]]]   
          
        time = len(days_dict)
        
        particleCount= np.tile(np.zeros([360,720]),(time,1,1)) #Output shape (time, width, height) of zeroes matrices
        
        for key in days_dict:
            
            
            x=[]
            y=[]
            k = days_dict[key]
            for value in k: 
                
                #print(value)
                x.append(int(round((width/360.0) * (180 + value[0])))) #lat
                y.append(int(round((height/180.0) * (90 - value[1])))) #long
           
            for i,j in zip(x,y):
                particleCount[key][i,j] += 1
         
        
        hf.create_dataset('ParticleDensiy', data=particleCount)
        
        return particleCount







if __name__ == "__main__":
    
    
    parser = ArgumentParser(description="Program prints randomly sampled particles inside a bounding box")
    parser.add_argument("-lat", "--latitude",  type=float, help="Enter the latitude in arc degrees")
    parser.add_argument("-long", "--longititude", type=float, help="Enter the longititude in arc degrees")
    parser.add_argument("-box", "--boundingbox", type=int, help="Enter the length of a half-side of the bounding box")
    parser.add_argument("-size", "--samplesize", type=float, help="Enter the number of samples you want to randomly sample")
    args = parser.parse_args()
    bpc = BoundedParticleCount(args.latitude,args.longititude,args.boundingbox,args.samplesize)
    bboundParticles,p_idx = bpc.boundedBoxRandomParticles()
    pcountGridMap = bpc.convertParticlesToParticlesCount(bboundParticles)
    

