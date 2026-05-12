import kagglehub
import shutil
import os

# Download latest version
print("Downloading competition data...")
path = kagglehub.competition_download('2024444-d')

print("Path to competition files:", path)

target_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
os.makedirs(target_dir, exist_ok=True)

for item in os.listdir(path):
    s = os.path.join(path, item)
    d = os.path.join(target_dir, item)
    if os.path.isdir(s):
        if not os.path.exists(d):
            shutil.copytree(s, d)
        else:
            print(f"Directory {d} already exists. Skipping.")
    else:
        shutil.copy2(s, d)

print(f"Data successfully copied to {target_dir}")
