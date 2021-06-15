import os
import sys
import argparse

network_file_path = "./network/predict.py"
model_path = "./model2019_07"

def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-s', type=str, required=True,
                            dest='src_file',    help="msa file in .a3m format")
    parser.add_argument('-t1', type=str, required=True,
                            dest='target_file1',    help="target file with .npz extension")
    parser.add_argument('-t2', type=str, required=True,
                            dest='target_file2',    help="target file with .npy extension")
    args = parser.parse_args()
    
    return args

args = get_args()

src_path = args.src_file
target_path1 = args.target_file1
target_path2 = args.target_file2


if os.path.exists(src_path) and (not os.path.exists(target_path1)):
    print(src_path)
    os.system("python " + network_file_path + " -m " + model_path + " " + src_path + " " + target_path1 + " " + target_path2)
else:
    print("Src file not found or target file already present")