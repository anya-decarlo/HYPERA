 from pathlib import Path
    working_dir = Path.cwd()
    data_path = working_dir / 'BBBC039_metadata/'
    with open(data_path/ 'training.txt') as file:
        training_names = set(file.read().split('.png\n'))
    with open(data_path/ 'validation.txt') as file:
        validation_names = set(file.read().split('.png\n'))
    with open(data_path/ 'test.txt') as file:
        test_names = set(file.read().split('.png\n'))



train_files = [data_path / name for name in train_names]
val_files = [data_path / name for name in validation_names]
test_files = [data_path / name for name in test_names]

from pathlib import Path
    working_dir = Path.cwd()
    data_path = working_dir / 'BBBC039_metadata/'
    with open(data_path/ 'training.txt') as file:
        training_names = set(file.read().split('.png\n'))
    with open(data_path/ 'validation.txt') as file:
        validation_names = set(file.read().split('.png\n'))
    with open(data_path/ 'test.txt') as file:
        test_names = set(file.read().split('.png\n'))



#take the pathlib and use it to make 
train_ds = Dataset(data=train_files, transform=train_transforms)
val_ds = Dataset(data=val_files, transform=val_transforms)