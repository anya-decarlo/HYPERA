import os
import numpy as np 
import sys 


file_path_test = "Users/anyadecarlo/HYPERA/BBBC039_metadata/test.txt"


def file_to_list(file_path_test): 
    try: 
        with open(file_path_test, "r") as file: 
            lines = file.read().splitlines()
        return lines 
    except FileNotFoundError:
        print("non at {file_path}")
    return None


test_list = file_to_list(file_path_test)

print(test_list)


exists = os.path.exists(file_path_test)
print(exists)

print(file_path_test)

