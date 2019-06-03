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
#distance between two point using the longitude and latitude
from math import cos, asin, sqrt
def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295     #Pi/180
    a = 0.5 - cos((lat2 - lat1) * p)/2 + cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2
    return 12742 * asin(sqrt(a)) #2*R*asin...
#######################################################
# objective of this function is to crop the image with the specified size
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

##  normalized the value between 0 to 1 of the hsl value obtained above
new_elevation_normalised = (new_image - (np.min(new_image))) / (np.max(new_image) - np.min(new_image))

import skimage.measure
import sklearn.feature_extraction as sfe

# we get the cell from the image with the 15*15 window sliding over whole image
#it takes the mean value of the 15*15 window

new_elevation_mean = skimage.measure.block_reduce(new_elevation_normalised,(15,15), np.mean)
plt.imshow(new_elevation_mean) 


 ## to make the sea level to zero level
new_elevation_mean[new_elevation_mean < 0.1] = 0


# crop the image in smaller size
filtered_elevation = filtered_image(new_elevation_mean,40,40,315,140)
plt.hist(filtered_elevation.ravel())

# to obtain the list of unique values

my_list=np.unique(filtered_elevation)

# to comvert the list of value upto two decimal place
my_formatted_list = [ '%.2f' % elem for elem in my_list ]

# to plot the images
plt.xlabel('Normalized elevation')
plt.ylabel('Number of cells')
plt.title('Filtered elevation histogram')
plt.hist(filtered_elevation.ravel())
plt.hist(filtered_elevation.ravel(), edgecolor='black', linewidth=1.2)
plt.savefig('Hist_filtered_elevation.png')
np.unique(filtered_elevation)

import pandas as pd
my_list_elevation=[]
for i in range(0, filtered_elevation.shape[0]):
    for j in range(0, filtered_elevation.shape[1]):
        my_list_elevation.extend([filtered_elevation[i][j]])
data1_elevation = pd.DataFrame(my_list_elevation)

cm = plt.cm.get_cmap('jet')

# Get the histogramp
Y,X = np.histogram(data1_elevation, 25)
x_span = X.max()-X.min()
C = [cm(((x-X.min())/x_span)) for x in X]
plt.xlabel('Normalized elevation')
plt.ylabel('Number of cells')
plt.title('Filtered elevation histogram')
plt.bar(X[:-1],Y,color=C,width=X[1]-X[0])
plt.savefig('Hist_filtered_elevation.png')
plt.show()



#(88,45)is brest and (141,296) is riken .the location in the original image

#(48,5)is brest and (101,256) is riken .the location in the filtered image

plt.imshow(filtered_elevation, cmap='gray')
plt.title('Cropped elevation image')
plt.savefig('Cropped elevation image.png')
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

#knock out the pixel value less than 0.15 to make them as zero population
p[p < 0.15] = 0
plt.hist(p.ravel())
np.unique(p)

#new image with mean of the sliding window of 15*15 
new_population_mean = skimage.measure.block_reduce(p,(15,15), np.mean)
plt.imshow(new_population_mean)  

# to crop the image
filtered_population = filtered_image(new_population_mean,40,40,315,140)
# plot the image
plt.hist(filtered_population.ravel())
plt.imshow(filtered_population, cmap='gray')
plt.title('Cropped population image')
plt.savefig('Cropped population image.png')
plt.show()
################################################################
plt.xlabel('Normalized population')
plt.ylabel('Number of cells')
plt.title('Filtered population histogram')
plt.hist(filtered_population.ravel())
plt.hist(filtered_population.ravel(), edgecolor='black',color = "gray", linewidth=2)
plt.savefig('Hist_Filtered population.png')
np.unique(filtered_elevation)


##############################################################

import pandas as pd
my_list_population=[]
for i in range(0, filtered_population.shape[0]):
    for j in range(0, filtered_population.shape[1]):
        my_list_population.extend([filtered_population[i][j]])
data1_population = pd.DataFrame(my_list_population)

cm = plt.cm.get_cmap('gray')

# Get the histogramp
Y,X = np.histogram(data1_population, 25,edgecolor='black')
x_span = X.max()-X.min()
C = [cm(((x-X.min())/x_span)) for x in X]
plt.xlabel('Normalized population')
plt.ylabel('Number of cells')
plt.title('Filtered population histogram')
plt.bar(X[:-1],Y,color=C,width=X[1]-X[0])
plt.savefig('Hist_filtered_population.png')
plt.show()



import math

## objective of this function is to calculate the elevation angle betwen current
# node and next node     
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
        normalised_slop=(max - x) / (max - min)# to convert it between 0 to 1
    else:
        normalised_slop=1
    return angle_degree,normalised_slop,adjacent_cell_elevation


def build_neighbor_pos_list(pos, n_row, n_col):
    """
    Use list comprehension to create a list of all positions in the cell's in neighborhood.
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

## to calculate the human_population
def calcaute_population(pos):
    a,b =pos
    return Human_population[a][b]

# to calulcate total population arround the cell       
def total_population_nbr_cell(l):
    list_pop=[]
    for a in l:
        list_pop.append(calcaute_population(a))
    return sum(list_pop)



#########calculate the number of zombies  to move to the next cell
def population_propogate(nbr_population,total_population,Zombies,elevation):
    return int(math.ceil((nbr_population/total_population)*elevation*Zombies))



##  to return the zombies count from the current cell
def get_zombies(pos):
    x,y = pos
    return zombies[x][y]
    
    

from scipy.ndimage.interpolation import shift

#zombies= np.random.randint(5, size=(3, 3,15))
# to shift the zombies per day 
def shift_zombies_per_day():
    for i in range(zombies.shape[0]):
        for j in range(zombies.shape[1]):
             lst = zombies[i][j]
             zombies[i][j] = shift(lst, 1, cval=0)


# this function updates the zombies
# it take the current position, next position and number of the zombies going 
# to be pased to the next position as per defined formula
def update_zombies(zombies_position,zombies_prop,next_zombies_position):
    count=zombies_prop
    zombies_next = get_zombies(next_zombies_position)
    zombies_current = get_zombies(zombies_position)
    partial_zombies= np.zeros(zombies_current.shape).astype(int)# empty array to store the temprory zombies information 
    #print("zombies going to propagate",zombies_prop)
    #print('sum of zombies here',sum(zombies_current))
    for i in range(*zombies_current.shape):## divide the zombies as per their ratio in the subcells(count of zombies as per their age)
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
    #add the zombies in  the next cell
    zombies[next_zombies_position]=[x + y for x, y in zip(partial_zombies, zombies_next)]
    #substract  the zombies from  the current cell
    zombies[zombies_position]=[y - x for x, y in zip(partial_zombies, zombies_current)]


## this part basically deals with the zombies and humam fight########
# objective of the function is to kill the humans 10 times of zombies count
    
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
        
# objective of the function is to kills the zombies 10 times of human count       
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

#set function to update the human with new population     
def update_human_population(pos,value):
    Human_population[pos] = value    
    
    
### to store the information about the nodes as a nested dictionary  
def _nested_update(document, key, value,run):
    if document != {}:
        if key in document.keys():
            if run == 0:
                document[key].update(value[key])
        run = run - 1
        for dict_key, dict_value in document.items():
            _nested_update(document=dict_value, key=key, value=value,run=run)
    return document

##########################################################################
### to store the information of zombies movement  in a  dictionary     
def _diction_update(document, key, value):
    if document != {}:
        if key in document.keys():
                document[key].update(value[key])
        else:
            document[key] = (value[key])
    return document

########################################################################
    
#to search for the values in the nested  dictionary diction

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
                        

######## to perform the step 1####################

def Zombies_movement_step1(Nodes,run):## list of node where zombies are located currently and the count of number of days
    tree={}  ## empty dictionary to store zombies positiion
    List=[]
    track_count = run+1
    run = run + 1
    shift_zombies_per_day() # shift the zombies and make them a day older
    for node in Nodes:
        Zombies = get_zombies(node)
        if(sum(Zombies)>0):# check if the zombies are greater than zero
            tree.update({node: {}})
            x,y = node #unpack the tuple
            # get the neighbouring cells
            cells = build_neighbor_pos_list((x,y),filtered_elevation.shape[0], filtered_elevation.shape[1])
            # calculate the total popoulation in neighbouring cells
            total_population = total_population_nbr_cell(cells)
            #itertate over the populations
            for positions in cells:
                #calculate the angle slope and hieght between the cells
                angle,slope,height_level = calcaulte_elevation(node,positions)
                # to calculate the nulber of population in current cell
                nbr_population = calcaute_population(positions)
                if angle < 10.0 and total_population > 0 and nbr_population>0 and height_level>0:
                    tree[node].update({positions: {}})
                    List.extend([positions])
                    #print((nbr_population,total_population,sum(Zombies),slope))
                    # calculate amount of zombies propogate in the cell
                    zombies_prop = int(math.ceil(population_propogate(nbr_population,total_population,sum(Zombies),slope)))
                    #update the zombies by adding and substracing them 
                    update_zombies(node,zombies_prop,positions)
            List = list(sorted(set(List)))
            #_nested_update(diction, node, tree,run)
            #update the zombies information in the dictionary
            _diction_update(diction, node, tree)
    return List,track_count


# step 2 function to kills the humans
def zombies_and_human_step2(Nodes):
    for node in Nodes:
        zombies_kills_human(node)
# step 3 function to kills the zombies        
def human_killing_zombies_step3(Nodes):
    for node in Nodes:
        Human_kills_zombies(node)


#############basic_prep_for_data#########################        
diction={} # empty diciton to store the cells inforamtion with zombies movemement
start_pos=(100,255)  # starting posiiton of the city riken
End_pos =(47,4) # end position of city brest
diction.setdefault(start_pos, {}) # to uodate the diction with the starting cells
List=[(100,255)]
track =-1 # to track the number of days 


####Store the information about human population in numpy array
#each 15*15 cell contain 3000*15*15 people if pixel is bright
Human_population= np.ceil((filtered_population * 3000 * 15 * 15)).astype(int)

#####Store the information about the zombies in numpy array
# size of the zombies is (234, 322, 15)##########
# zombies information are stored at depth of 15 in order to store the count of them

zombies = np.zeros((filtered_elevation.shape[0], filtered_elevation.shape[1],15)).astype(int)

zombies_2d_array = np.sum(zombies, axis=-1)

zombies[100][255][0]=Human_population[100][255]

## this function contains set of function to be performed in each day 
# with step1 step2 and step3 respectively
# it will return the list indicating where zombies moved int the previous 
#movement and number of days

def execute_code(List,track):
    List,track = Zombies_movement_step1(List,track)
    zombies_and_human_step2(List)
    human_killing_zombies_step3(List)
    print(track)
    return List,track

zombies_directory ='/home/rudresh/Documents/MSC2/DataScience_Graph/Challenge_3/Zombies_movement/'

# run the while loop unless the zombies reached to brest location(47,4)
while(((47,4) in List) == False):
    List,track=execute_code(List,track)
    #List = Short_Distance_finder(List)
    # convert from 3d array to 2d array with the count of zombies in each cell
    zombies_2d_array = np.sum(zombies, axis=-1)
    plt.imshow(zombies_2d_array, cmap='Reds')
    plt.imshow(filtered_elevation, cmap='gray',alpha =0.5)
    plt.savefig(zombies_directory+'zombies_population'+str(track)+'.png')
    print('no of days',track)
    print('length of list is',len(List))
    

from collections import Mapping
# Empty directed graph
import networkx as nx
# function is to draw the graph
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
    return G
##########################################################################
    
G=draw_our_graph()

# to calculate the path length
print(nx.dijkstra_path_length(G,(100,255),(47,4)))

degree_centrality = nx.betweenness_centrality(G)  
    
#################################################################################

#from scipy.spatial import distance
#def Short_Distance_finder(target,nodeList):
#    list2={}
#    for a in target:
#        dists = [distance.sqeuclidean(a, i) for i in nodeList]
#        min_dist=min(dists)
#        list2.update({a:min_dist })
#    All_node_List = sorted(list2, key=list2.get, reverse=False)
#    return All_node_List[:100]
#    
#from scipy.spatial import distance
#
#def Short_Distance_finder(nodeList):
#    list2={}
#    End_pos =(47,4)
#    #node_list=[(48, 4), (49, 5), (50, 6), (50, 7), (49, 8), (50, 9), (51, 10), (51, 11), (52, 12), (51, 13), (52, 14), (53, 15), (52, 16), (53, 17), (52, 18), (51,19), (51, 20), (50, 21), (50, 22), (51, 23), (52, 24), (51, 25), (51, 26), (52, 27), (52, 28), (52, 29), (53, 30), (53, 31), (54, 32), (53, 33),(53, 34), (53, 35), (53, 36), (54, 37), (55, 38), (56, 39), (55, 40), (54, 41), (53, 42), (53, 43), (54, 44), (54, 45), (54, 46), (53, 47), (54, 48),(53, 49), (53, 50), (53, 51), (52, 52), (52, 53), (51, 54), (51, 55), (52, 56), (52, 57), (53, 58), (53, 59), (53, 60), (54, 61), (53, 62), (52, 63),(53, 64), (52, 65), (51, 66), (50, 67), (49, 68), (50, 69), (51, 70), (50, 71), (51, 72), (52, 73), (51, 74), (52, 75), (51, 76), (52, 77), (53, 78),(54, 79), (54, 80), (53, 81), (54, 82), (54, 83), (55, 84), (56, 85), (55, 86), (54, 87), (54, 88), (55, 89), (55, 90), (55, 91), (55, 92), (56, 93),(56, 94), (57, 95), (58, 96), (57, 97), (58, 98), (59, 99), (60, 100), (61, 101), (62, 102), (61, 103), (62, 104), (63, 105), (64, 106), (64, 107),(64, 108), (63, 109), (63, 110), (63, 111), (62, 112), (62, 113), (61, 114), (60, 115), (59, 116), (58, 117), (58, 118), (59, 119), (60, 120), (61,121), (62, 122), (62, 123), (61, 124), (60, 125), (59, 126), (59, 127), (60, 128), (60, 129), (60, 130), (59, 131), (60, 132), (59, 133), (58,134), (59, 135), (58, 136), (59, 137), (60, 138), (61, 139), (62, 140), (62, 141), (63, 142), (62, 143), (63, 144), (62, 145), (63, 146), (64,147), (64, 148), (64, 149), (64, 150), (63, 151), (64, 152), (65, 153), (66, 154), (67, 155), (68, 156), (69, 157), (70, 158), (71, 159), (72,160), (73, 161), (74, 162), (75, 163), (76, 164), (77, 165), (78, 166), (79, 167), (80, 168), (81, 169), (82, 170), (83, 171), (84, 172), (85,173), (86, 174), (87, 175), (88, 176), (89, 177), (90, 178), (91, 179), (92, 180), (93, 181), (94, 182), (95, 183), (96, 184), (97, 185), (98,186), (98, 187), (99, 188), (99, 189), (99, 190), (100, 191), (101, 192), (102, 193), (101, 194), (100, 195), (100, 196), (100, 197), (100, 198),(100, 199), (101, 200), (102, 201), (101, 202), (100, 203), (99, 204), (98, 205), (97, 206), (98, 207), (97, 208), (98, 209), (97, 210), (96, 211),(97, 212), (97, 213), (98, 214), (97, 215), (96, 216), (95, 217), (96, 218), (96, 219), (95, 220), (95, 221), (95, 222), (95, 223), (96, 224), (96,225), (96, 226), (96, 227), (97, 228), (98, 229), (98, 230), (99, 231), (98, 232), (99, 233), (99, 234), (100, 235), (100, 236), (101, 237), (102,238), (101, 239), (101, 240), (101, 241), (102, 242), (102, 243), (101, 244), (101, 245), (101, 246), (102, 247), (101, 248), (101, 249), (101,250), (101, 251), (102, 252), (102, 253), (101, 254)]
#    for nodes in nodeList:
#        #dists = [distance.sqeuclidean(End_pos, i) for i in nodes]
#        dist=distance.sqeuclidean(End_pos, nodes)
#        #min_dist=min(dists)
#        list2.update({nodes:dist })
#    All_node_List = sorted(list2, key=list2.get, reverse=False)
#    return All_node_List[:100]
       
#    
##execute the step2 here
#import operator
#def parse_step1_short_distance(Nodes,run,previous_stage_nodes):
#    tree_dict={}
#    List=[]
#    track_count = run+1
#    run = run + 1
#    shift_zombies_per_day()
#    for node in Nodes:
#        Zombies = get_zombies(node)# to get rid of 15 days old zombies since they wont move
#        if(sum(Zombies)>0):
#            tree= tree_dict
#            tree.update({node: {}})
#            list2={}
#            x,y = node
#            cells = build_neighbor_pos_list((x,y),filtered_elevation.shape[0], filtered_elevation.shape[1])
#            total_population = total_population_nbr_cell(cells)
#            global All_node_List
#            global two_stages_nodes
#            two_stages_nodes =previous_stage_nodes
#            print('here is two stage node',two_stages_nodes)
#            for positions in cells:
#                angle,slope,height_level = calcaulte_elevation(node,positions)
#                nbr_population = calcaute_population(positions)
#                if angle < 10.0 and total_population > 0 and nbr_population>0 and height_level>0:
#                    x1,y1 = positions
#                    dist = math.hypot(47 - x1, 4 - y1)
#                    list2.update({positions:dist })
#                    #print((nbr_population,total_population,sum(Zombies),slope))
#                    zombies_prop = int(math.ceil(population_propogate(nbr_population,total_population,sum(Zombies),slope)))
#                    update_zombies(node,zombies_prop,positions)
#                    tree[node].update({positions: {}})
#            min_distance_node =min(list2.items(), key=operator.itemgetter(1))[0]
#            get_zombies(min_distance_node)
#            All_node_List = sorted(list2, key=list2.get, reverse=False)# sort dict in decresing order
#            two_stages_nodes.extend(All_node_List)
#            print('here is also two stage node',two_stages_nodes)
#            List.extend([min_distance_node])
#            _nested_update(diction, node, tree,run)
#            print(node,List)
#    return List,track_count,All_node_List[1:-1],two_stages_nodes[1:-1]
#
#diction={}
#start_pos=(100,255)
#End_pos =(47,4)
#diction.setdefault(start_pos, {})
#List=[(100,255)]
#track =-1
#
#####Store the information about human population in numpy array
#Human_population= np.ceil((filtered_population * 3000 * 15 * 15)).astype(int)
#
#zombies = np.zeros((filtered_elevation.shape[0], filtered_elevation.shape[1],15)).astype(int)
#
#zombies_2d_array = np.sum(zombies, axis=-1)
#
#zombies[100][255][0]=Human_population[100][255]
#
#track_previous_node=[(100,255)]
#
#i=0
#
###### this is an algorithm with two step back tracking##########################
#
#
#def execute_code_short_distance(List,track,previous_node):
#    global i
#    global track_previous_node
#    global track_second_stage
#    print('prevoisu node',previous_node)
#    List,track,ALL_nodes,two_stage_no = parse_step1_short_distance(List,track,previous_node)
#    print('this is list',List)
#    #print('this is all node',ALL_nodes)
#    print('this is double node',two_stage_no)
#    if (len(List)!=0):
#        i=0
#        zombies_and_human_step2(List)
#        print('i am killing human')
#        human_killing_zombies_step3(List)
#        track_previous_node=ALL_nodes
#        track_second_stage=two_stage_no
#        print('this list you are looking for',track_previous_node)
#        print('this is the double list',track_second_stage)
#    else:
#        
#        print('#########the obtained list were empty so check other value################')
#        List=[track_second_stage[i]]
#        i+=1
#        print("considere this List",List)
#        track-=1
#        execute_code_short_distance(List,track,previous_node=track_second_stage)
#    print(track)
#    return List,track
#        
#while(len(nested_lookup((47,4), diction))==0):
#    List,track=execute_code_short_distance(List,track,track_previous_node)
#        
#
####### to crop the image from centre#####
#
#
#def crop_center(img,cropx,cropy):
#    y,x = img.shape
#    startx = x//2-(cropx//2)
#    starty = y//2-(cropy//2)    
#    return img[starty:starty+cropy,startx:startx+cropx]  
#
#
################################recursive dictionary###############
#def recursive_items(dictionary):
#    for key, value in dictionary.items():
#        if type(value) is dict:
#            yield (key, value)
#            yield from recursive_items(value)
#        else:
#            yield (key, value)
#
#nodes_list=[]
#
#for key, value in recursive_items(diction):
#    nodes_list.extend([key])
#    
############################################################3
#    #print nested dictionary in the ordered way
#    
#def pretty(d, indent=0):
#   for key, value in d.items():
#      print('\t' * indent + str(key))
#      if isinstance(value, dict):
#         pretty(value, indent+1)
#      else:
#         print('\t' * (indent+1) + str(value))