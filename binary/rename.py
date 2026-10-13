import os

folder = "results_reverse"  # 改成你的文件夹路径

for filename in os.listdir(folder):
    if filename.startswith("urw_") and filename.endswith(".csv"):
        new_name = "urw_reverse_" + filename[len("urw_"):]

        old_path = os.path.join(folder, filename)
        new_path = os.path.join(folder, new_name)

        os.rename(old_path, new_path)
        print(f"{filename} -> {new_name}")