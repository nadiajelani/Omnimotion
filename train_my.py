import os
import shutil
import random

output_root = "/Users/nadiajelani/Documents/GitHub/omni/dataset"
source_root = "/Users/nadiajelani/Documents/GitHub/omni/converted_dataset"

train_folder = os.path.join(output_root, "train")
val_folder = os.path.join(output_root, "val")
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

all_sequences = []
for ball_folder in ["Ball 1", "Ball 2", "Ball 3"]:
    for drop_folder in ["Drop 1", "Drop 2", "Drop 3", "Drop 4", "Drop 5"]:
        seq_path = os.path.join(source_root, ball_folder, drop_folder)
        if os.path.exists(seq_path):
            all_sequences.append(seq_path)

random.shuffle(all_sequences)

split_index = int(len(all_sequences) * 0.8)
train_sequences = all_sequences[:split_index]
val_sequences = all_sequences[split_index:]

def move_images(sequences, output_dir, prefix):
    for i, seq in enumerate(sequences):
        seq_folder = os.path.join(output_dir, f"{prefix}{i+1:03d}")
        os.makedirs(seq_folder, exist_ok=True)
        for img_file in sorted(os.listdir(seq)):
            if img_file.endswith((".png", ".jpg", ".bmp")):
                shutil.move(os.path.join(seq, img_file), os.path.join(seq_folder, img_file))

move_images(train_sequences, train_folder, "train_")
move_images(val_sequences, val_folder, "val_")

print("✅ Dataset split successfully!")
