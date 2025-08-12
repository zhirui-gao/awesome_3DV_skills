import os
import time
import json
import shutil


# blenderproc run examples/datasets/abc_dataset/main_mv.py  examples/datasets/abc_dataset/00000006/00000006_d4fe04f0f5f84b52bd4f10e4_trimesh_001.obj examples/datasets/abc_dataset/output
# blenderproc vis hdf5 examples/datasets/abc_dataset/output/*.hdf5 --save examples/datasets/abc_dataset/output_rgb/






def create_synthetic_dataset(dataset_dir, cad_name, output_dataset_dir):
    print("cad_name:", cad_name)
    # if os.path.exists(os.path.join(output_dataset_dir, cad_name, 'transforms_train.json')):
    #     print(cad_name + " already exist")
    #     return
    obj_names = os.listdir(dataset_dir)
    # Filter obj names to only get the one matching the cad_name prefix
    filtered_obj_names = [name for name in obj_names if name.startswith(cad_name)]
    print(filtered_obj_names)
    output_dir = os.path.join(output_dataset_dir, cad_name)
    
    obj_path = os.path.join(dataset_dir, filtered_obj_names[0])
    print("Processing " + obj_path)
    os.makedirs(output_dir, exist_ok=True)
    num_images = 50

    command = "blenderproc run examples/datasets/abc_dataset/main_mv.py " + obj_path + " " + output_dir + \
              " "+str(num_images)+" train"
    print(command)
    os.system(command)
    time.sleep(3)

    # hdf5_files = os.listdir(out_dir_hdf5)
    # for hdf5_file in hdf5_files:
    #     hdf5_file_path = os.path.join(out_dir_hdf5, hdf5_file)
    #     # load hdf5 file
    #     import h5py
    #     with h5py.File(hdf5_file_path, 'r') as f:
    #         print(f.keys())
    
    # command = "blenderproc vis hdf5 " + out_dir_hdf5 + "/train/*.hdf5 --save " + output_dir + "/train"
    # print(command)
    # os.system(command)
    # time.sleep(3)

    # shutil.rmtree(out_dir_hdf5)


dataset_dir = "/media/gzr/955be20b-af2b-4597-83f8-8585ff878672/ABC_dataset/ABC-NEF_Edge/groundtruth/obj"

output_dataset_dir = "/media/gzr/955be20b-af2b-4597-83f8-8585ff878672/ABC_dataset/render"

names = os.listdir(dataset_dir)
cad_names = [name.split('_')[0] for name in names]

for cad_name in cad_names:
    create_synthetic_dataset(dataset_dir, cad_name, output_dataset_dir)


# env: py38
# python render_ABC.py




