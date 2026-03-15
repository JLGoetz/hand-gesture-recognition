import numpy as np
import os

# 1. Get the directory where this script is located
base_dir = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(base_dir, 'training_data.npy')

def load_and_clean():
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        return

    # Load the data
    data = np.load(DATA_FILE, allow_pickle=True).item()

    # Display status
    print("\n--- Current Dataset Status ---")
    for label, samples in data.items():
        print(f"Gesture: {label:10} | Samples: {len(samples)}")

    # Interactive cleaning
    target = input("\nEnter a gesture name to DELETE (or press Enter to skip): ").upper()
    
    if target in data:
        confirm = input(f"Are you sure you want to delete ALL {len(data[target])} samples of '{target}'? (y/n): ")
        if confirm.lower() == 'y':
            del data[target]
            np.save(DATA_FILE, data)
            print(f"'{target}' removed successfully.")
    elif target != "":
        print(f"Gesture '{target}' not found in dataset.")

if __name__ == "__main__":
    load_and_clean()