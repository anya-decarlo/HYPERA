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

path.cwd()






from pathlib import Path
    working_dir = Path.cwd()
    # monai.config.print_config()
    data_path = working_dir / 'BBBC039_metadata/'
    with open(data_path/ 'training.txt') as file:
        training_names = set(file.read().split('.png\n'))
    with open(data_path/ 'validation.txt') as file:
        validation_names = set(file.read().split('.png\n'))
    with open(data_path/ 'test.txt') as file:
        test_names = set(file.read().split('.png\n'))
    # print(f"Training names: {training_names}")
    # print(f"Validation names: {validation_names}")
    # print(f"Test names: {test_names}")


    