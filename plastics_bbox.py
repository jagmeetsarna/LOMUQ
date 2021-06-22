from argparse import ArgumentParser
import h5py
import pandas as pd
import os
import random   
import matplotlib.pyplot as plt
import math



os.chdir(r"D:\TOPIOS Data\data\twentyone") #Please change the directory to where your particle dataset is


filename = "particles.h5"

data = h5py.File(filename, "r")

# for key in data.keys():
#     print(key)




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

def printmap(lat,long,bbox,sample_size):
    #Initializaing
    px = data['p_x'].value
    py = data['p_y'].value
    px_attrs = data['p_x'].attrs
    py_attrs = data['p_y'].attrs
    extends = (float(px_attrs['min']), float(px_attrs['max']), float(py_attrs['min']), float(py_attrs['max']))
    
    #Timesteps
    t= [1,2] 
    
    rlist=[]
    for j in t:
        arr = []
       
        for i in range(len(px)):
            
            arr.append([px[i][j], py[i][j]])
    
        bound_box = get_bounding_box(lat,long,bbox)
        inside_boundingbox=[]
    
        for i in arr:
        
            if bound_box.lat_min <= i[0] and i[0] <= bound_box.lat_max and bound_box.lon_min <= i[1] and i[1] <= bound_box.lon_max :
                
                inside_boundingbox.append([i[0], i[1]])
    
    #Random sampling
    
        #random.seed(9001)
             
        if not rlist:
            
            rlist=[]
            for i in range(sample_size):
            
                r=random.randint(1,100)
                if r not in rlist: rlist.append(r)
    
    
        inside_boundingbox_random = []
        inside_boundingbox_random =[inside_boundingbox[i] for i in rlist]
    
        df = pd.DataFrame(inside_boundingbox_random, columns=['lat', 'long'])
        
    
        
        # Scatterplot 1 - father heights vs son heights with darkred square markers
        #plt.figure()
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8), frameon=False,facecolor="#222222")
        ax.axis('off')
    
        ax.scatter(df['long'] , df['lat'],color='white')
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        
        ax.set_xlim([extends[0], extends[1]])
        ax.set_ylim([extends[2], extends[3]])
        
        #plt.scatter(df['long'] , df['lat'])
    
        # Show your plot
        plt.savefig('time'+ str(j) + '.png', dpi=200)
        plt.show()


if __name__ == "__main__":
    
    
    parser = ArgumentParser(description="Program prints randomly sampled particles inside a bounding box")
    parser.add_argument("-lat", "--latitude",  type=float, help="Enter the latitude in arc degrees")
    parser.add_argument("-long", "--longititude", type=float, help="Enter the longititude in arc degrees")
    parser.add_argument("-box", "--boundingbox", type=int, help="Enter the length of a half-side of the bounding box")
    parser.add_argument("-size", "--samplesize", type=int, help="Enter the number of samples you want to randomly sample")
    args = parser.parse_args()
    printmap(args.latitude,args.longititude,args.boundingbox,args.samplesize)
    print(df.shape)

    
    
