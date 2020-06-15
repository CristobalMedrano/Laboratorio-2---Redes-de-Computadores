#-*- coding: utf-8 -*-
#AUTHOR: Cristóbal Nicolás Medrano Alvarado (19.083.864-1)
#DATE: 19/06/2020
#LABORATORY 2: CONVOLUCIÓN 2D

# IMPORTS
import imageio
import os.path

# CONSTANTS 
# GLOBAL VARIABLES
# CLASSES
class Image:

    def __init__(self, filename, matrix, width, height):
        self.__filename = filename
        self.__matrix = matrix
        self.__width = width
        self.__height = height

    @property
    def filename(self):
        return self.__filename
    
    @filename.setter
    def set_filename(self, filename):
        self.__filename = filename

    @property
    def matrix(self): 
        return self.__matrix
    
    @matrix.setter
    def set_matrix(self, matrix):
        self.__matrix = matrix

    @property
    def width(self): 
        return self.__width
    
    @width.setter
    def set_width(self, width):
        self.__width = width

    @property
    def height(self): 
        return self.__height
    
    @height.setter
    def set_heigth(self, height):
        self.__height = height

# FUNCTIONS
def is_valid_image_file(filename):
    """ Check if it's a valid image file.
    
    Returns True if it's valid, False if it's not.
    
    Parameters:
    ----------
    filename : string
        Input image filename. 

    Returns:
    -------
    Status: boolean
        True if it's valid, False if it's not.
   """
    if not os.path.exists(filename):
        return False
    elif filename[-4:] != ".bmp":
        return False
    else:
        return True

def read_image(filename):
    matrix = imageio.imread(filename)
    width = matrix.shape[0]
    height = matrix.shape[1]
    return Image(filename, matrix, width, height)

def save_image(filename, matrix):
    imageio.imwrite(f"salida_{filename}", matrix)

# MAIN
def main():
    """ Main function of program """
    print("Laboratorio 2 - Convolución 2D")
    filename = "lena512.bmp"
    if is_valid_image_file(filename):
        original_image = read_image(filename)
        save_image(original_image.filename, original_image.matrix)
        

# MAIN
main()

# REFERENCES
#https://pypi.org/project/imageio/