#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 02:41:23 2019

@author: rudresh
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 01:40:55 2019

@author: rudresh
"""

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
plt.imshow(filtered_population, cmap='gray')
plt.show()

####Store the information about human population in numpy array
Human_population= np.ceil((filtered_population * 3000 * 15 * 15))

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

def calcaute_population(pos):
    a,b =pos
    return Human_population[a][b]

# to calulcate total population arround the cell       
def total_population_nbr_cell(l):
    list_pop=[]
    for a in l:
        list_pop.append(calcaute_population(a))
    return sum(list_pop)
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


import itertools

from scipy import misc
from scipy.sparse.dok import dok_matrix
from scipy.sparse.csgraph import dijkstra


# Create a flat color image for graph building:
img = filtered_population

img_elevation = filtered_elevation


# Defines a translation from 2 coordinates to a single number
def to_index(x, y):
    return x * img.shape[1] + y


# Defines a reversed translation from index to 2 coordinates
def to_coordinates(index):
    return math.ceil(index / img.shape[1]), index % img.shape[1]


# A sparse adjacency matrix.
# Two pixels are adjacent in the graph if both are painted.
adjacency = dok_matrix((img.shape[0] * img.shape[1],
                        img.shape[0] * img.shape[1]), dtype=bool)

# The following lines fills the adjacency matrix by
directions = list(itertools.product([0, 1, -1], [0, 1, -1]))
for i in range(1, img.shape[0]-1):
    for j in range(1, img.shape[1]-1):
        if not img[i, j]:
            continue
        cells = build_neighbor_pos_list((i,j),filtered_elevation.shape[0], filtered_elevation.shape[1])
        total_population = total_population_nbr_cell(cells)
        for positions in cells:
                x,y =positions
                angle,slope,height_level = calcaulte_elevation((i,j),positions)
                nbr_population = calcaute_population(positions)
                if angle < 10.0 and total_population > 0 and nbr_population>0 and height_level>0:
                    adjacency[to_index(i, j),to_index(x,y)] = True

# We chose two arbitrary points, which we know are connected
source = to_index(100,255)
target = to_index(47,4)



# Compute the shortest path between the source and all other points in the image
_, predecessors = dijkstra(adjacency, directed=False, indices=[source],
                           unweighted=True, return_predecessors=True)

# Constructs the path between source and target
pixel_index = target
pixels_path = []
while pixel_index != source:
    pixels_path.append(pixel_index)
    pixel_index = predecessors[0, pixel_index]


# The following code is just for debugging and it visualizes the chosen path
import matplotlib.pyplot as plt

for pixel_index in pixels_path:
    i, j = to_coordinates(pixel_index)
    img_elevation[i, j] = 0

plt.imshow(img_elevation)
plt.show()

from scipy.spatial import distance

def Short_Distance_finder(List):
    list2={}
    node_list=[(48, 4), (49, 5), (50, 6), (50, 7), (49, 8), (50, 9), (51, 10), (51, 11), (52, 12), (51, 13), (52, 14), (53, 15), (52, 16), (53, 17), (52, 18), (51,19), (51, 20), (50, 21), (50, 22), (51, 23), (52, 24), (51, 25), (51, 26), (52, 27), (52, 28), (52, 29), (53, 30), (53, 31), (54, 32), (53, 33),(53, 34), (53, 35), (53, 36), (54, 37), (55, 38), (56, 39), (55, 40), (54, 41), (53, 42), (53, 43), (54, 44), (54, 45), (54, 46), (53, 47), (54, 48),(53, 49), (53, 50), (53, 51), (52, 52), (52, 53), (51, 54), (51, 55), (52, 56), (52, 57), (53, 58), (53, 59), (53, 60), (54, 61), (53, 62), (52, 63),(53, 64), (52, 65), (51, 66), (50, 67), (49, 68), (50, 69), (51, 70), (50, 71), (51, 72), (52, 73), (51, 74), (52, 75), (51, 76), (52, 77), (53, 78),(54, 79), (54, 80), (53, 81), (54, 82), (54, 83), (55, 84), (56, 85), (55, 86), (54, 87), (54, 88), (55, 89), (55, 90), (55, 91), (55, 92), (56, 93),(56, 94), (57, 95), (58, 96), (57, 97), (58, 98), (59, 99), (60, 100), (61, 101), (62, 102), (61, 103), (62, 104), (63, 105), (64, 106), (64, 107),(64, 108), (63, 109), (63, 110), (63, 111), (62, 112), (62, 113), (61, 114), (60, 115), (59, 116), (58, 117), (58, 118), (59, 119), (60, 120), (61,121), (62, 122), (62, 123), (61, 124), (60, 125), (59, 126), (59, 127), (60, 128), (60, 129), (60, 130), (59, 131), (60, 132), (59, 133), (58,134), (59, 135), (58, 136), (59, 137), (60, 138), (61, 139), (62, 140), (62, 141), (63, 142), (62, 143), (63, 144), (62, 145), (63, 146), (64,147), (64, 148), (64, 149), (64, 150), (63, 151), (64, 152), (65, 153), (66, 154), (67, 155), (68, 156), (69, 157), (70, 158), (71, 159), (72,160), (73, 161), (74, 162), (75, 163), (76, 164), (77, 165), (78, 166), (79, 167), (80, 168), (81, 169), (82, 170), (83, 171), (84, 172), (85,173), (86, 174), (87, 175), (88, 176), (89, 177), (90, 178), (91, 179), (92, 180), (93, 181), (94, 182), (95, 183), (96, 184), (97, 185), (98,186), (98, 187), (99, 188), (99, 189), (99, 190), (100, 191), (101, 192), (102, 193), (101, 194), (100, 195), (100, 196), (100, 197), (100, 198),(100, 199), (101, 200), (102, 201), (101, 202), (100, 203), (99, 204), (98, 205), (97, 206), (98, 207), (97, 208), (98, 209), (97, 210), (96, 211),(97, 212), (97, 213), (98, 214), (97, 215), (96, 216), (95, 217), (96, 218), (96, 219), (95, 220), (95, 221), (95, 222), (95, 223), (96, 224), (96,225), (96, 226), (96, 227), (97, 228), (98, 229), (98, 230), (99, 231), (98, 232), (99, 233), (99, 234), (100, 235), (100, 236), (101, 237), (102,238), (101, 239), (101, 240), (101, 241), (102, 242), (102, 243), (101, 244), (101, 245), (101, 246), (102, 247), (101, 248), (101, 249), (101,250), (101, 251), (102, 252), (102, 253), (101, 254)]
    for nodes in List:
        dists = [distance.sqeuclidean(nodes, i) for i in node_list]
        min_dist=min(dists)
        list2.update({nodes:min_dist })
    All_node_List = sorted(list2, key=list2.get, reverse=False)
    return All_node_List[:6]