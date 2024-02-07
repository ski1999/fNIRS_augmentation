import os

path = "/home/ski1999/CNN_plv_hamv1_fold"
directory = os.listdir(path)
os.chdir(path)

for (path,dir,files) in os.walk(path):
    for file in files:
        filename_list = file.split("_")
        if len(filename_list) == 4:
            print(file)
            new_path = os.path.join(path, file)
            os.remove(new_path)
        elif len(filename_list) == 8:
            print(file)
            new_path = os.path.join(path, file)
            os.remove(new_path)
        elif filename_list[2] == 'PCC':
            print(file)
            new_path = os.path.join(path, file)
            os.remove(new_path)                        