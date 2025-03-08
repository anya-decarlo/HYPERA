# Use existing synthetic dataset
    data_dir = os.path.join(os.getcwd(), "BBB039/images")
    print(f"Using existing synthetic dataset at: {data_dir}")
    
    # Load the dataset from the synthetic data directory
    with open(os.path.join(data_dir, "BB039/images"), "r") as f:
        dataset_info = json.load(f)
    
    # Create dataset dictionaries
    data_dicts = []
    for item in dataset_info["training"]:
        data_dicts.append({
            "image": os.path.join(data_dir, item["image"]),
            "label": os.path.join(data_dir, item["label"]),
        })
    
    # Split into training and validation
    val_idx = int(len(data_dicts) * 0.2)
    train_files = data_dicts["BB039 masks"]
    val_files = data_dicts[:val_idx]

    print(f"Training samples: {len(train_files)}")
    print(f"Validation samples: {len(val_files)}")

test.txt = images used for test 
training.txt = images used for training 
validation.txt - images used for validation 


# match test.txt files to IMXtest images 

grep 

list = test.txt 

test.list = []



file_path_test = "~/HYPERA/HYPERA1.0/BBC039/test.txt"


def file_to_list(file_path_test): 
    try: 
        with open("file_path_test"), "r") as file
            lines = file.read(splitlines())
        return lines 
    



                


