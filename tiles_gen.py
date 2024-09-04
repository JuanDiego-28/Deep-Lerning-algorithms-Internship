#!/bin/bash 

"""

author: jdieg@ 

"""

# imprting the libs

from pathlib import Path
import re
import json
import numpy as np
import pyvips
import cv2
from rasterio.features import shapes
from shapely.geometry import Polygon
import geopandas as gp
from matplotlib import pyplot as plt
from IPython.display import clear_output
#------------------------------------------#
# work with openslide                      # 
#------------------------------------------#
from openslide import open_slide
import os
import openslide

from skimage.color import lab2rgb
import imageio



#------------------------------------------------------------------
#Def all function to create grid, cropp , filter and store tiles  
#------------------------------------------------------------------

# define the grid , get slide points  

def generate_grid(grid_size, slide):
    """Generate shapely shaped of tiles with side size (grid_size)
    of the slide"""
    width, height= slide.width, slide.height
    n_rows= int(np.ceil(height)/grid_size)
    n_cols= int(np.ceil(width)/grid_size)
    XleftOrigin = 0
    XrightOrigin = 0 + grid_size
    YtopOrigin = 0
    YbottomOrigin = 0 + grid_size
    tiles=[]
    for i in range(n_cols):
        y_top= YtopOrigin
        y_bottom= YbottomOrigin
        for j in range(n_rows):
            tiles.append(Polygon([(XleftOrigin, y_top), (XrightOrigin, y_top), (XrightOrigin, y_bottom), (XleftOrigin, y_bottom)])) # (top left), (top right), (bottom right), (bottom left)
            y_top = y_top + grid_size
            y_bottom = y_bottom + grid_size
        XleftOrigin = XleftOrigin + grid_size
        XrightOrigin = XrightOrigin + grid_size
    return tiles

def slice_points(tiles):
    """generated a list of lists with the upper left coordinate of each tile"""
    slice_points = []
    for tile in tiles:
        minx, miny, maxx, maxy= tile.bounds
        slice_points.append((int(minx), int(maxy)))
    return slice_points

def tiles_points(grid_size, slide):
    """wrapper for tile shapes and cutting coordinates"""
    tiles = generate_grid(grid_size, slide)
    points = slice_points(tiles)
    return tiles, points

############################################################################################################

# atomatize the cutting proces 

def cropp ( left , top , GRID_SIZE , slide_img) : 
    '''
    This function crop the slides in the index it recieves 

    INPUT:
        tile left and top  indexs 
    OUTPUT: 
        tiles  
    ''' 

    tile = slide_img.crop(left, top, GRID_SIZE, GRID_SIZE)

    return tile

# discard borders 
def cleaner (cutting_points, slide_img):
    """
    filtering the border images 

    INPUT: 
        List with cutting points od the slide
    OUTPUT:
        Filtered cutting points 
    """
    width, height= slide_img.width, slide_img.height

    new_width = (width // 512) * 512  # // is floor division
    new_height = (height // 512) * 512

    for n in  cutting_points:
        if n[1] == new_height:
            ind = list(cutting_points).index(n)
            del cutting_points[ind]
    return cutting_points

#  save tiles 

def store_tiles ( cutting_points , slide_img ,GRID_SIZE ) :

    '''
    This function return tiles as np.array 

    INPUT:
        List with the tiles index
    OUTPUT: 
        return tiles as np.array 
    ''' 

    save_tile = []
    
    # loop over cutting points 
    
    for n in range (0,len(cutting_points)):  

        left, top = cutting_points[ n ]

        #print('shw tile in postion', (left, top))

        tile = cropp (left , top , GRID_SIZE , slide_img )


        # converting to np.array
        tile_array = np.ndarray(buffer=tile.write_to_memory(),
                   dtype=np.uint8,
                   shape=[tile.height, tile.width, tile.bands])
        
        # discarding alpha clannel
        tile_array = tile_array[:,:,:1]

        save_tile.append(tile_array)
            
    return save_tile

##############################################################################################################################################

# function to separate background and data 

def saving_filtered_tiles (save_t ,  path2 , path3 ,GRID_SIZE = 512):
    
    '''
    This function stores filtered tiles as images.png 

    INPUT:
        list with np.array tiles 
    OUTPUT: 
        stores the tiles 
    ''' 

    # Umbral para determinar si el tile contiene datos relevantes
    threshold = 0.10
    num = 0 
    for tile in save_t: 
    
    # use adpatative threshold 
        binary_tile = cv2.adaptiveThreshold(tile , 255 ,cv2.ADAPTIVE_THRESH_MEAN_C , cv2.THRESH_BINARY_INV, 127, 5 )
        non_zero_pixels = cv2.countNonZero(binary_tile)
        total_pixels = GRID_SIZE * GRID_SIZE 

        # if amount of pixels not zero is  grater than treshold the store the tile
        if non_zero_pixels / total_pixels > threshold:
        
            # save fig
            
            plt.imshow(tile, vmin=0, vmax=255)
            plt.axis('off')

            # store tiles 
            ima_save = path2 + 'tile_B.{:1d}.png'.format(num)
            #imageio.imwrite(ima_save, tile)
            plt.savefig(ima_save)
            #print((tile.shape))
            plt.close()

        elif non_zero_pixels / total_pixels <= threshold:
    
            # save fig 
            #tile = np.clip(tile.astype(np.uint8), 0, 255)
            plt.imshow(tile, vmin=0, vmax=255)
            plt.axis('off')

            # store tiles 
            ima_save = path3 + 'tile.{:1d}.png'.format(num)
            plt.savefig(ima_save)
            plt.close()
            #print('guardado no data')

        num +=1

def saving_tiles_nofiltered (save_t ,path1 , GRID_SIZE = 512):
    '''
    This function stores all tiles as images.png 

    INPUT:
        list with np.array tiles 
    OUTPUT: 
        stores the tiles 
    ''' 
    
    # Umbral para determinar si el tile contiene datos relevantes
    num = 0 
    for tile in save_t: 
    
    # use adpatative threshold 
        plt.imshow(tile, vmin=0, vmax=255)
        plt.axis('off')

        # store tiles 
        ima_save = path1 + 'tile.{:1d}.png'.format(num)
        plt.savefig(ima_save)
        plt.close()
        num +=1 



#===================================================
# Running the functions 
#===================================================


path = '/media/jdiego/0fcbc4f2-ab1a-4855-894c-f5ca933658e2/jdiego/inter/pics/'    #replace with  path were slides are
path2 = '/media/jdiego/0fcbc4f2-ab1a-4855-894c-f5ca933658e2/jdiego/inter/tiny_B/data_color/'   #replace with path to store filtered tiles
path3 = '/media/jdiego/0fcbc4f2-ab1a-4855-894c-f5ca933658e2/jdiego/inter/tiny_B/background_color/'   #replace with path to store not filtered tiles


ima = 'tiny.png'  # replace with slide name 
# define grid size for tiling
GRID_SIZE= 512 


# search slide 
slide_path = path + ima

slide_img = pyvips.Image.new_from_file(slide_path)

# Display the image to verify how it looks
slide_img_array = np.array(slide_img)
plt.figure(figsize=(8, 6))
plt.imshow(slide_img_array, cmap='gray')
plt.title('Slide Image')
plt.axis('off')
plt.show()


# reading as grayscale
slide_img = slide_img.colourspace("b-w")    

# generating all tiles of the slide and the corresponding cutting points
tiles, cutting_points = tiles_points(GRID_SIZE, slide_img)

# clean 
cutting_points_c = cleaner (cutting_points , slide_img)

# return tiles as np.array 

save_t = store_tiles ( cutting_points_c , slide_img , GRID_SIZE)

# save filtered tiles 
saving_filtered_tiles(save_t ,  path2 , path3 )

# save all tiles 
#saving_tiles_nofiltered(save_t , path1)

print("Finish")

