# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 17:37:59 2017

@author: cbothore
"""
# https://fr.wikipedia.org/wiki/Windows_bitmap

# A first method to parse a BMP file
# It is a binary file
# Import a module to convert bytes from binary files 
# to H (unsigned short, 2 bytes), I (unsigned int, 4 bytes)
import struct

input_filename="/home/rudresh/Documents/MSC2/DataScience_Graph/Challenge_3/population-density-map.bmp"
input_elevation="/home/rudresh/Documents/MSC2/DataScience_Graph/Challenge_3/elevation_wrt_population.bmp"

bmp = open(input_filename, 'rb') # open a binary file
print('-- First part of the header, information about the file (14 bytes)')
print('Type:', bmp.read(2).decode())
print('Size: %s' % struct.unpack('I', bmp.read(4)))
print('Reserved 1: %s' % struct.unpack('H', bmp.read(2)))
print('Reserved 2: %s' % struct.unpack('H', bmp.read(2)))
offset=struct.unpack('I', bmp.read(4))
print('Image start after Offset: %s' % offset)

print('-- Second part of the header, DIB header, bitmap information header (varying size)')
print('The size of this DIB Header Size: %s' % struct.unpack('I', bmp.read(4)))
print('Width: %s' % struct.unpack('I', bmp.read(4)))
print('Height: %s' % struct.unpack('I', bmp.read(4)))
print('Colour Planes: %s' % struct.unpack('H', bmp.read(2)))
pixel_size=struct.unpack('H', bmp.read(2))
print('Bits per Pixel: %s' % pixel_size)
print('Compression Method: %s' % struct.unpack('I', bmp.read(4)))
print('Raw Image Size: %s' % struct.unpack('I', bmp.read(4)))
print('Horizontal Resolution: %s' % struct.unpack('I', bmp.read(4)))
print('Vertical Resolution: %s' % struct.unpack('I', bmp.read(4)))
print('Number of Colours: %s' % struct.unpack('I', bmp.read(4)))
print('Important Colours: %s' % struct.unpack('I', bmp.read(4)))

# At this step, we have read 14+40 bytes
# As offset[0] = 54, from now, we will read the BMP content
# You have to read each pixel now, and do what you have to do
# First pixel is bottom-left, and last one top-right
# .........
bmp.close()


# Another method to parse a BMP image
# To manipulate imageIf you want to work with image data in Python, 
# numpy is the best way to store and manipulate arrays of pixels. 
# You can use the Python Imaging Library (PIL) to read and write data 
# to standard file formats.

# Use PIL module to read file
# http://pillow.readthedocs.io/en/latest/
from PIL import Image

import numpy as np
im = Image.open(input_filename)
im.show()

# This modules gives useful informations
width=im.size[0]
heigth=im.size[1]
colors = im.getcolors(width*heigth)
print('Nb of different colors: %d' % len(colors))
# To plot an histogram
from matplotlib import pyplot as plt
def hexencode(rgb):
    r=rgb[0]
    g=rgb[1]
    b=rgb[2]
    return '#%02x%02x%02x' % (r,g,b)

for idx, c in enumerate(colors):
    plt.bar(idx, c[0], color=hexencode(c[1]))

plt.show()
# We have 32 different colors in this image
# We can see that we have "only" 91189 black pixels able to stop zombies 
# but we have a large majority of dark ones slowing their progression

# With the image im, let's generate a numpy array to manipulate pixels
p = np.array(im) 

print(p.shape)
# a result (3510, 4830, 3) means (rows, columns, color channels)
# where 3510 is the height and 4830 the width

# to get the Red value of pixel on row 3 and column 59
p[3,59][0]

# How to get the coordinates of the green and red pixels where 
# (0,0) is top-left and (width-1, height-1) is bottom-right
# In numpy array, notice that the first dimension is the height, 
# and the second dimension is the width. That is because, for a numpy array, 
# the first axis represents rows (our classical coord y), 
# and the second represents columns (our classical x).

# First method
# Here is a double loop (careful, O(n²) complexity) to parse the pixels from
# (0,0) top-left and (heigth-1, width-1) is bottom-right
for y in range(heigth):
    for x in range(width):
        # p[y,x] is the coord (x,y), x the colum, and y the line
        # As an exemple, we search for the green and red pixels
        # p[y,x] is an array with 3 values
        # We test if there is a complete match between the 3 values 
        # from both arrays p[y,x] and np.array([0,255,0])
        # to detect green pixels
        if (p[y,x] == np.array([0,255,0])).all():
            print("Coordinates (x,y) of the green pixel: (%s,%s)" % (str(x),str(y)))
            # Coordinates (x,y) of the green pixel: (4426,2108)
        if (p[y,x] == np.array([255,0,0])).all():
            print("Coordinates (x,y) of the red pixel: (%s,%s)" % (str(x),str(y)))
            # Coordinates (x,y) of the red pixel: (669,1306)

# Here is a more efficient method to get the location of the green and red pixels
mask = np.all(p == (0, 255, 0), axis=-1)
z = np.transpose(np.where(mask))
print("Coordinates (x,y) of the green pixel: (%d,%d)" % (z[0][1],z[0][0]))
mask = np.all(p == (255, 0, 0), axis=-1)
z = np.transpose(np.where(mask))
print("Coordinates (x,y) of the red pixel: (%d,%d)" % (z[0][1],z[0][0]))


# Now we have the source and the target positions of our zombies
# we could convert our RGB image into greyscale image to manipulate
# only 1 value for the color and deduce more easily the density of
# population
grayim = im.convert("L")
grayim.show()
colors = grayim.getcolors(width*heigth)
print('Nb of different colors: %d' % len(colors))
# With the image im, let's generate a numpy array to manipulate pixels
p = np.array(grayim) 
# plot the histogram. We still have a lot of dark colors. Just to check ;-)
plt.hist(p.ravel())

# from gray colors to density
density = p/255.0
# plot the histogram. We still have a lot of dark colors. Just to check ;-)
plt.hist(density.ravel())

# We can use the gray 2D array density to create our graph
# Gray colors density[y,x] range now from 0 (black) to 1 (white)
# density[0,0] is top-left pixel density
# and density[heigth-1,width-1] is bottom-right pixel

#############################################################################
#############################################################################
########### the code ########################################################
#############################################################################
#############################################################################
### here we deal with the elevation image which we have resized



im_ele= Image.open(input_elevation)

width_ele=im_ele.size[0]
heigth_ele=im_ele.size[1]

colors_ele = im_ele.getcolors(width_ele*heigth_ele)
print('Nb of different colors: %d' % len(colors_ele))

p_ele = np.array(im_ele)

############## Dsitance between two points################
from math import cos, asin, sqrt
def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295     #Pi/180
    a = 0.5 - cos((lat2 - lat1) * p)/2 + cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2
    return 12742 * asin(sqrt(a)) #2*R*asin...
#######################################################
def filtered_image(img,start_x,start_y,end_x,end_y):
    return img[start_y:start_y+end_y,start_x:start_x+end_x]  


##################elevation code##################
# to convert the thr rgb value in the hls format
# it scale the value in between 0.43 to 1 depeding on blue to red pixel respectively
        
import colorsys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread(input_elevation)
img = img[ : , : , 0:3] # keep only r, g, b channels  

new_image = np.zeros((img.shape[0], img.shape[1]))

for y_pos in range(img.shape[0]):
    for x_pos in range (img.shape[1]):

        color = img[y_pos, x_pos]
        r,g,b = color
        h, _, _ = colorsys.rgb_to_hls(r, g, b) 
        new_image[y_pos, x_pos] = 1.0 - h

plt.imshow(new_image, cmap='gray')
plt.show()

p_elevation = np.array(new_image)
plt.hist(p_elevation.ravel())
np.unique(p_elevation)

## thus function normalized the value between 0 to 1
new_elevation_normalised = (new_image - (np.min(new_image))) / (np.max(new_image) - np.min(new_image))

# we get the cell from the image with the 15*15 window sliding over whole image
#it takes the mean value of the 15*15 window

import skimage.measure
import sklearn.feature_extraction as sfe

new_elevation_mean = skimage.measure.block_reduce(new_elevation_normalised,(15,15), np.mean)
plt.imshow(new_elevation_mean) 

new_elevation_mean[new_elevation_mean < 0.01] = 0 ## to make the sea level at zero level

filtered_elevation = filtered_image(new_elevation_mean,40,40,315,140)

#(88,45)is brest and (141,296) is riken .the location in the original image

#(48,5)is brest and (101,256) is riken .the location in the filtered image

plt.imshow(filtered_elevation, cmap='gray')
plt.show()

###****** First normalization:###############
#convert population image into the grayscale image

img_gray = im.convert("L")
G_gray = np.float32((np.array(img_gray)))

# reduce the scale in order to avoid useless calculate
#G_gray = G_gray[x1:x2,y1:y2]

#  Normalize the values of all elements between 0 and 1
g_normalised = (G_gray - (np.min(G_gray))) / (np.max(G_gray) - np.min(G_gray))
p = np.array(g_normalised)
p[p < 0.10] = 0
plt.hist(p.ravel())
np.unique(p)

new_population_mean = skimage.measure.block_reduce(p,(15,15), np.mean)
plt.imshow(new_population_mean)  

filtered_population = filtered_image(new_population_mean,40,40,315,140)

####Store the information about human population in numpy array
Human_population= np.ceil((filtered_population * 3000 * 15 * 15))


import math

      
def calcaulte_elevation(current_pos,next_pos):
    a,b =current_pos
    x,y = next_pos
    max = 0.176327  # valueof tan10
    min=0
    current_cell_elevation = filtered_elevation[a][b]
    adjacent_cell_elevation= filtered_elevation[x][y]
    difference_elevation= adjacent_cell_elevation - current_cell_elevation
    maximum_level = 4810 ## it is equivqlent to 1 on our gray scale
    #we know the value the distance between the centre of two cells of image is 1 km 
    # calculate the inverse tan with the help of elevation and distance between the centre
    #two cells
    elevation = difference_elevation * maximum_level
    if abs(a-x)==1 and abs(b-y)==1:
        x=elevation/(15*math.sqrt(2)*1000)
        angle_radian = np.arctan(x)
        angle_degree = math.degrees(angle_radian)
    else:
        x=elevation/(15*1000)
        angle_radian = np.arctan(x)
        angle_degree = math.degrees(angle_radian)
    if(angle_degree>0):
        normalised_slop=(max - x) / (max - min)
    else:
        normalised_slop=1
    return angle_degree,normalised_slop,adjacent_cell_elevation


def build_neighbor_pos_list(pos, n_row, n_col):
    """
    Use list comprehension to create a list of all positions in the cell's Moore neighborhood.
    Valid positions are those that are within the confines of the domain (n_row, n_col)
    and not the same as the cell's current position.

    :param pos: cell's position; tuple
    :param n_row: maximum width of domain; integer
    :param n_col: maximum height of domain; integer
    :return: list of all valid positions around the cell
    """
    # Unpack the tuple containing the cell's position
    r, c = pos
    l = [(r+i, c+j)
         for i in [-1, 0, 1]
         for j in [-1, 0, 1]
         if 0 <= r + i < n_row
         if 0 <= c + j < n_col
         if not (j == 0 and i == 0)]

    return l

## to calculate the human_populatio
def calcaute_population(pos):
    a,b =pos
    return Human_population[a][b]

# to calulcate total population arround the cell       
def total_population_nbr_cell(l):
    list_pop=[]
    for a in l:
        list_pop.append(calcaute_population(a))
    return sum(list_pop)



#########calculate the zombies  to move to next cell
def population_propogate(nbr_population,total_population,Zombies,elevation):
    return int(math.ceil((nbr_population/total_population)*elevation*Zombies))


   
#####Store the information about the zombies in numpy array
# size of the zombies is (234, 322, 15)##########
zombies = np.zeros((filtered_elevation.shape[0], filtered_elevation.shape[1],15))

zombies_2d_array = np.sum(zombies, axis=-1)


def get_zombies(pos):
    x,y = pos
    return zombies[x][y]
    
    

from scipy.ndimage.interpolation import shift

#zombies= np.random.randint(5, size=(3, 3,15))

def shift_zombies_per_day():
    for i in range(zombies.shape[0]):
        for j in range(zombies.shape[1]):
             lst = zombies[i][j]
             zombies[i][j] = shift(lst, 1, cval=0)


    
def update_zombies(zombies_position,zombies_prop,next_zombies_position):
    count=zombies_prop
    zombies_next = get_zombies(next_zombies_position)
    zombies_current = get_zombies(zombies_position)
    partial_zombies= np.zeros(zombies_current.shape).astype(int)
    #print("zombies going to propagate",zombies_prop)
    #print('sum of zombies here',sum(zombies_current))
    for i in range(*zombies_current.shape):
        if (zombies_current[i]!=0):
            partial_zombies[i] = int(math.floor(zombies_prop*(zombies_current[i]/sum(zombies_current))))
            
    value = math.floor(count-sum(partial_zombies))
    #print(partial_zombies)
    #print(zombies_current)
    while value!=0:
        x= np.random.randint(15)
        if(partial_zombies[x]<zombies_current[x]):
            partial_zombies[x]+=1
            value-=1
    print('i got it')
    zombies[next_zombies_position]=[x + y for x, y in zip(partial_zombies, zombies_next)]
    zombies[zombies_position]=[y - x for x, y in zip(partial_zombies, zombies_current)]


## this part basically deals with the zombies and humam fight########
    
def zombies_kills_human(positions):
    zombies_current = get_zombies(positions)
    human_tobe_killed = 10 * sum(zombies_current)
    humans= calcaute_population(positions)
    count = humans - human_tobe_killed
    #print("human to get killed",count)
    if (count <=0):
        update_human_population(positions,0)
        zombies[positions][0]+= humans 
    else:
        update_human_population(positions,count)
        zombies[positions][0]+= human_tobe_killed 
        
        
def Human_kills_zombies(positions):
    zombies_current = get_zombies(positions)
    partial_zombies= np.zeros(zombies_current.shape).astype(int)
    zombies_tobe_killed = 10*Human_population[positions]
    #print('sum of the zombies',sum(zombies_current))
    #print("no of zombies to be killed",zombies_tobe_killed)
    count= sum(zombies_current)-zombies_tobe_killed
    if(count<=0):
        zombies[positions]=partial_zombies
    else:
        for i in range(*zombies_current.shape):
            if (zombies_current[i]!=0):
                partial_zombies[i] = int(math.floor(zombies_tobe_killed*(zombies_current[i]/sum(zombies_current))))
        value = zombies_tobe_killed-sum(partial_zombies)
        while value!=0 and sum(partial_zombies)<sum(zombies_current):
            x= np.random.randint(15)
            if(partial_zombies[x]<zombies_current[x]):
                partial_zombies[x]+=1
                value-=1
        zombies[positions]=[y - x for x, y in zip(partial_zombies, zombies_current)]

#to update the human with new population        
def update_human_population(pos,value):
    Human_population[pos] = value


### to store the information of dictionary  




def _nested_update(document, key, value,run):
    if isinstance(document, dict):
        if key in document.keys():
            if run == 0:
                document[key].update(value[key])
        for dict_key, dict_value in document.items():
            run = run - 1
            _nested_update(
                document=dict_value, key=key, value=value, run=run
            )
    return document

######## to perform the step 1####################

def parse_step1(Nodes,run):
    tree_dict={}
    List=[]
    track_count = run+1
    run = run + 1
    shift_zombies_per_day()
    for node in Nodes:
        Zombies = get_zombies(node)# to get rid of 15 days old zombies since they wont move
        if(sum(Zombies)>0):
            tree= tree_dict
            tree.update({node: {}})
            x,y = node
            cells = build_neighbor_pos_list((x,y),filtered_elevation.shape[0], filtered_elevation.shape[1])
            List.extend(cells)
            total_population = total_population_nbr_cell(cells)
            for positions in cells:
                angle,slope,height_level = calcaulte_elevation(node,positions)
                nbr_population = calcaute_population(positions)
                if angle < 10.0 and total_population > 0 and nbr_population>0 and height_level>0:
                    tree[node].update({positions: {}})
                    #print((nbr_population,total_population,sum(Zombies),slope))
                    zombies_prop = int(math.ceil(population_propogate(nbr_population,total_population,sum(Zombies),slope)))
                    update_zombies(node,zombies_prop,positions)
            _nested_update(diction, node, tree,run)
    return List,track_count




def zombies_and_human_step2(Nodes):
    for node in Nodes:
        zombies_kills_human(node)
        
def human_killing_zombies_step3(Nodes):
    for node in Nodes:
        Human_kills_zombies(node)
        

from collections import Mapping
# Empty directed graph
import networkx as nx

def draw_our_graph():
    G = nx.DiGraph()
    # Iterate through the layers
    q = list(diction.items())
    while q:
        v, d = q.pop()
        for nv, nd in d.items():
            G.add_edge(v, nv)
            if isinstance(nd, Mapping):
                q.append((nv, nd))
    
    np.random.seed(8)
    pos = nx.spring_layout(G,k=1,iterations=20)
    color_map = []
    for node in G:
       if node == start_pos:
           color_map.append('red')
       else: color_map.append('green')      
    nx.draw(G,pos, with_labels=True,node_size= 800, node_color = color_map)
    plt.show()
##########################################################################
#to search for the values in the dictionary diction

from collections import defaultdict

def nested_lookup(key, document, wild=False, with_keys=False):
    """Lookup a key in a nested document, return a list of values"""
    if with_keys:
        d = defaultdict(list)
        for k, v in _nested_lookup(key, document, wild=wild, with_keys=with_keys):
            d[k].append(v)
        return d
    return list(_nested_lookup(key, document, wild=wild, with_keys=with_keys))


def _nested_lookup(key, document, wild=False, with_keys=False):
    """Lookup a key in a nested document, yield a value"""
    if isinstance(document, dict):
            for k, v in document.items():
                if key == k or (wild and key.lower() in k.lower()):
                    if with_keys:
                        yield k, v
                    else:
                        yield v
                if isinstance(v, dict):
                    for result in _nested_lookup(key, v, wild=wild, with_keys=with_keys):
                        yield result
diction={}
start_pos=(100,255)
End_pos =(47,4)
diction.setdefault(start_pos, {})
List=[(100,255)]
track =-1

####Store the information about human population in numpy array
Human_population= np.ceil((filtered_population * 3000 * 15 * 15)).astype(int)

zombies = np.zeros((filtered_elevation.shape[0], filtered_elevation.shape[1],15)).astype(int)

zombies_2d_array = np.sum(zombies, axis=-1)

zombies[100][255][0]=Human_population[100][255]

while(len(nested_lookup((47,4), diction))==0):
    List,track = parse_step1(List,track)
    print(track)
    zombies_and_human_step2(List)
    human_killing_zombies_step3(List)



###### to crop the image from centre#####


def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]  