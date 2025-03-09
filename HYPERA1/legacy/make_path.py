import os
import numpy as np 
import sys 



file_path_test = "~/HYPERA/HYPERA1.0/BBC039/test.txt"




def file_to_list(file_path_test): 
    try: 
        with open(file_path_test, "r") as file: 
            lines = file.read().splitlines()
        return lines 
    except FileNotFoundError:
        print("non at {file_path}")
    return None