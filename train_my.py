import os
import shutil
import random

# ✅ Define output_root first
output_root = "/Users/nadiajelani/Documents/GitHub/omni/dataset/"  # Make sure this is correct

# ✅ Define source root (your converted dataset location)
source_root = "/Users/nadiajelani/Documents/GitHub/omni/converted_dataset"

# ✅ Create train/val folders
train_folder = os.path.join(output_root, "train")
val_folder = os.path.join(output_root, "val")
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)


# ✅ Collect all sequences
# ✅ Add debug messages
all_sequences = []
for ball_folder in ["Ball 1", "Ball 2", "Ball 3"]:
    for drop_folder in ["Drop 1", "Drop 2", "Drop 3", "Drop 4", "Drop 5"]:
        seq_path = os.path.join(source_root, ball_folder, drop_folder)
        if os.path.exists(seq_path):
            print(f"✅ Found sequence: {seq_path}")  # Debugging
            all_sequences.append(seq_path)
        else:
            print(f"⚠️ Missing sequence: {seq_path}")  # Debugging

# ✅ Shuffle for randomness
random.shuffle(all_sequences)

# ✅ Split into train (80%) and validation (20%)
split_index = int(len(all_sequences) * 0.8)
train_sequences = all_sequences[:split_index]
val_sequences = all_sequences[split_index:]

# ✅ Function to move images into sequences
print("🚀 Starting to move images to train folder...")

def move_images(sequences, output_dir, prefix):
    for i, seq in enumerate(sequences):
        seq_folder = os.path.join(output_dir, f"sequence_{prefix}{i+1:03d}")
        os.makedirs(seq_folder, exist_ok=True)
        print(f"📂 Saving to: {seq_folder}")  # Debug print

        for img_file in sorted(os.listdir(seq)):
            if img_file.endswith((".png", ".jpg", ".bmp")):
                print(f"📷 Moving {img_file}...")  # Debug before copying
                shutil.copy(os.path.join(seq, img_file), os.path.join(seq_folder, img_file))

    print(f"✅ Moved {len(sequences)} sequences to {output_dir}")

# ✅ Organize dataset into train/val folders
move_images(train_sequences, train_folder, "train_")
move_images(val_sequences, val_folder, "val_")

print("✅ Dataset successfully split into train/val!")
