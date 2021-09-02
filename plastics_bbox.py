import os
from pathlib import Path
from argparse import ArgumentParser
import h5py
import pandas as pd
import os
import random   
import matplotlib.pyplot as plt
import math
import numpy as np
import imageio

datadir_path = '/data/LOMUQ'
resultdir_path = '/data/LOMUQ/jssarna'
# datadir_path = 'F:\Lomuq Data'
# resultdir_path = 'F:\Lumoq Results'
pathlist = Path(datadir_path).rglob('*.*')
paths=[]
filesDict={}

for path in pathlist:
    
    
    path_in_str = str(path)
    paths = path_in_str.split('/')
    
    try:
        
        filesDict[paths[-2]].append(paths[-1])
    except KeyError:
        
        filesDict[paths[-2]] = [paths[-1]]





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
    
    def __init__(self,lat,long,bbox,sample_size,px,py,key):
        self.lat = lat
        self.long = long
        self.bbox = bbox
        self.sample_size = sample_size
        self.px = px
        self.py = py
        self.key = key
        
    
    def boundedBoxRandomParticles(self):

        #extends = (float(px_attrs['min']), float(px_attrs['max']), float(py_attrs['min']), float(py_attrs['max']))
        bound_box = get_bounding_box(self.lat,self.long,self.bbox)
        #bound_box = get_bounding_box(5,64,500)
        inside_boundingbox=[]
        inside_boundingbox_random = []
        randomList=[]
        #Timesteps
        
        timeframe = len(hydrodynamic_U_data['uo'])
        
        arr = []
       
        for i in range(len(self.px)):
        
            if bound_box.lat_min <= self.px[i,0] and self.px[i,0] <= bound_box.lat_max and bound_box.lon_min <= self.py[i,0] and self.py[i,0] <= bound_box.lon_max :
                # print(px[i,0])
                inside_boundingbox.append([self.px[i,0],self.py[i,0]])
                arr.append(i)
        
                
        randomList.append(random.sample(arr,round((len(inside_boundingbox)-1)*self.sample_size))) #random sampling
       
        for k in range(timeframe):
            
            for i in randomList[0]:
                #print(k)
                inside_boundingbox_random.append([self.px[i,k],self.py[i,k],k])
                        
        return inside_boundingbox_random,randomList
    
    def convertParticlesToParticlesCount(self,inside_boundingbox_random,width,height):
        
        
        # hf = h5py.File(resultdir_path+"/"+"data_topios"+str(self.key)+".h5",'w')
       
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
                
                lat_bucket = value[0]+90 #Turning -lats to positive i.e range 0,180
                lat_bucket = lat_bucket * 100 #18,000 hundredth-degree increments of latitude (i.e. -90.00, -89.99, ... 89.99, 90.00)
                lat_bucket = int(round(lat_bucket/(100/2))) #Increment by 0.5 for 360 rows, therefore 50. #50 is dynamic
                
                long_bucket = value[1]+180
                long_bucket = long_bucket * 100
                long_bucket = int(round(long_bucket/(100/2)))
                

                #print(value)
                x.append(lat_bucket) #lat
                y.append(long_bucket) #long
            for i,j in zip(x,y):
                
                particleCount[key][i,j] += 1
         
        particleCount = np.flip(particleCount,1) #Because matrix indices are different.  Flipping the rows
        
        # hf.create_dataset('ParticleCount', data=particleCount)
        imageio.mimwrite(resultdir_path+"/"+"particleCount_"+str(self.key)+".gif", particleCount)
        return particleCount



if __name__ == "__main__":
    
    
    parser = ArgumentParser(description="Program prints randomly sampled particles inside a bounding box")
    parser.add_argument("-lat", "--latitude",  type=float, help="Enter the latitude in arc degrees")
    parser.add_argument("-long", "--longititude", type=float, help="Enter the longititude in arc degrees")
    parser.add_argument("-box", "--boundingbox", type=int, help="Enter the length of a half-side of the bounding box")
    parser.add_argument("-size", "--samplesize", type=float, help="Enter the number of samples you want to randomly sample")
    args = parser.parse_args()
    particleCountList=[]
    for key in filesDict:
        try:
            
            if int(key)>=40 and int(key)<=70:
                
                #print(key,'-->',filesDict)
            
                hydrodynamic_U = datadir_path +"/" + key + "/"+"hydrodynamic_U.h5"
                hydrodynamic_V = datadir_path +"/" + key + "/"+"hydrodynamic_V.h5"
                particles = datadir_path +"/" + key + "/"+"particles.h5"
                hydrodynamic_U_data = h5py.File(hydrodynamic_U , "r")
                
                width, height = np.shape(hydrodynamic_U_data['uo'][0])
                particle_data = h5py.File(particles, "r")
                
                print(width,height)
                
                px = particle_data['p_y'][()]
                py = particle_data['p_x'][()]
                
                bpc = BoundedParticleCount(args.latitude,args.longititude,args.boundingbox,args.samplesize,px,py,key)
                bboundParticles,p_idx = bpc.boundedBoxRandomParticles()
                particleCountList.append(bpc.convertParticlesToParticlesCount(bboundParticles,width,height))
        except ValueError:
            continue
    
    r,c = np.shape(particleCountList[0][0])

    #Cropping the "action" area
    dataDict={}
    for i in range(len(particleCountList)):
        
        for j in range(len(particleCountList[i])):
            
            for rows in range(0,r-40,40):
                
                for columns in range(0,c-40,40):
                    
                    if(particleCountList[i][j][rows:rows+40,columns:columns+40].sum()>0):
                        
                        dataDict[(i,j,rows,columns)] = particleCountList[i][j][rows:rows+40,columns:columns+40].sum()
    
    df = pd.DataFrame(dataDict.keys())
    df.columns=['Data','day','rows','columns']

    minRow =  df['rows'].min()   
    maxRow =  df['rows'].max()    
    minCol =  df['columns'].min()   
    maxCol =  df['columns'].max() 
    
    particleCountList = np.asarray(particleCountList)
    particleCountList = particleCountList[...,minRow:maxRow+40,minCol:maxCol+40]
    
    
    print(minRow,maxRow,minCol,maxCol)
    print(np.shape(particleCountList))
    
    hf = h5py.File(resultdir_path+"/"+"particleCountList"+".h5",'w')
    hf.create_dataset('ParticleCount', data=particleCountList)
    
    particleCountList = h5py.File(resultdir_path+"/"+"particleCountList"+".h5", 'r')
    
    hydrodynamic_U_dataList=[]
    hydrodynamic_V_dataList=[]
    for key in filesDict:
        try:
            if int(key)>=40 and int(key)<=70:
                hydrodynamic_U = datadir_path +"/" + key + "/"+"hydrodynamic_U.h5"
                hydrodynamic_V = datadir_path +"/" + key + "/"+"hydrodynamic_V.h5"
                hydrodynamic_U_data = h5py.File(hydrodynamic_U , "r")
                hydrodynamic_V_data = h5py.File(hydrodynamic_V , "r")
                hydrodynamic_U_dataList.append(hydrodynamic_U_data['uo'][...,minRow:maxRow+40,minCol:maxCol+40])
                hydrodynamic_V_dataList.append(hydrodynamic_V_data['vo'][...,minRow:maxRow+40,minCol:maxCol+40])
        except ValueError:
            continue
    
    
    hf = h5py.File(resultdir_path+"/"+"hydrodynamic_U_dataList"+".h5",'w')
    hf.create_dataset('hydrodynamic_U', data=hydrodynamic_U_dataList)
    
    hf = h5py.File(resultdir_path+"/"+"hydrodynamic_V_dataList"+".h5",'w')
    hf.create_dataset('hydrodynamic_V', data=hydrodynamic_V_dataList)


