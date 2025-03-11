import os
import numpy as np

def rename_file(root):
    for filename in os.listdir(root):
        if 'L╨Æ' in filename:
            old_file_path = os.path.join(root, filename)

            # Define the new file name (you can customize this logic as needed)
            new_file_name = filename.replace('L╨Æ','')
            new_file_path = os.path.join(root, new_file_name)

            # Rename the file
            if os.path.exists(new_file_path):
                os.remove(new_file_path)

            os.rename(old_file_path, new_file_path)
            print(f"Renamed {filename} to {new_file_name}")
root=r'output_dataset\test\test_88'

string_size =32 #2048
number_of_user = 100
def generate_key():
    # Generate a list of 1024 binary numbers (0s and 1s)
    bs_list = {i: np.random.randint(0, 2, string_size, dtype=np.uint8) for i in range(1, number_of_user + 1)}
    # Save to a numpy file
    np.save("bs_list.npy", bs_list)
    print("Binary numbers saved as bs_list.npy")

def pad_arrays(a, b):
    max_len = max(len(a), len(b))
    a_padded = np.pad(a, (0, max_len - len(a)), mode='constant', constant_values=0)
    b_padded = np.pad(b, (0, max_len - len(b)), mode='constant', constant_values=0)
    return a_padded, b_padded

def dice_coefficient(a, b):
    intersection = np.sum(a * b)
    return 2 * intersection / (np.sum(a) + np.sum(b))

#generate_key()