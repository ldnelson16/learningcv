import os

folder_path = "images"
prefix = "luke"

for i in range(1, 101):
    old_name = f"Untitled-{i:03d}.jpg"
    new_name = f"{prefix}{i-101}.jpg"
    old_path = os.path.join(folder_path, old_name)
    new_path = os.path.join(folder_path, new_name)
    
    try:
        os.rename(old_path, new_path)
        print(f"Renamed {old_name} to {new_name}")
    except FileNotFoundError:
        print(f"File {old_name} not found.")
    except FileExistsError:
        print(f"File {new_name} already exists.")

prefix="parker"

for i in range(101, 201):
    old_name = f"Untitled-{i:03d}.jpg"
    new_name = f"{prefix}{i-101}.jpg"
    old_path = os.path.join(folder_path, old_name)
    new_path = os.path.join(folder_path, new_name)
    
    try:
        os.rename(old_path, new_path)
        print(f"Renamed {old_name} to {new_name}")
    except FileNotFoundError:
        print(f"File {old_name} not found.")
    except FileExistsError:
        print(f"File {new_name} already exists.")