import os
import shutil
import random

data_dir = "data"
train_dir = "training_data"
test_dir = "testing_data"

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

valid_pairs = []
for file in os.listdir(data_dir):
    if file.endswith(".wav"):
        base = file[:-4]
        tsv_file = f"{base}.tsv"
        if os.path.exists(os.path.join(data_dir, tsv_file)):
            valid_pairs.append(base)

# shuffle and split
random.shuffle(valid_pairs)
split_index = int(len(valid_pairs) * 0.7)
train_files = valid_pairs[:split_index]
test_files = valid_pairs[split_index:]

def copy_pair(base_name, dest_dir):
    for ext in [".wav", ".tsv"]:
        src = os.path.join(data_dir, base_name + ext)
        dst = os.path.join(dest_dir, base_name + ext)
        shutil.copy(src, dst)

# training
for base in train_files:
    copy_pair(base, train_dir)

# testing
for base in test_files:
    copy_pair(base, test_dir)

print(f"Done. {len(train_files)} files in training, {len(test_files)} in testing.")
