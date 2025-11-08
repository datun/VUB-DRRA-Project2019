import zipfile
import os
from shutil import rmtree


# with respect to the inputs that are passed from train.py
# extracts the contents of zip files to a temporary directory to be processed
def prep_files(root_path, extr_path, folder_path, label):
    new_extr_path = os.path.join(root_path,"extr\\"+ label)     # path of temporary dir
    os.makedirs(new_extr_path)
    with zipfile.ZipFile(extr_path,'r') as zip_ref:
        zip_ref.extractall(new_extr_path)   # extracting all files found within
    # if the folder_path is not None, that means the images are contained within
    # a path that is asked from the user. The lines will move files to root of temporary dir.
    if folder_path is not None:
        actual_path = os.path.join(new_extr_path, folder_path)
        for filename in os.listdir(actual_path):
            os.rename(os.path.join(actual_path, filename), os.path.join(new_extr_path, filename))
    # remove any remaining directories
    folders = [f.path for f in os.scandir(new_extr_path) if f.is_dir()]
    for i in folders:
        rmtree(i, ignore_errors=True)

    return new_extr_path

