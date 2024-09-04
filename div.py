#!/bin/bash 

"""
author: jdieg@ 
"""


import cv2
import pyvips
from openslide import open_slide

slide_path = '/home/jdiego/Documentos/S_intern/images_svs/KBMDT_1703T47576.svs'                                    # repalce with your path to the image  

slide_img = pyvips.Image.new_from_file(slide_path, access="sequential")

 
area = slide_img.crop(17000, 21000, 10000, 10000)     # location and area you want to cropp ()


area.write_to_file("/media/jdiego/0fcbc4f2-ab1a-4855-894c-f5ca933658e2/jdiego/inter/tiny1.png")                            # dir were you want to save the image + format 

