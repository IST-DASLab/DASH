import socket

from fineweb import get_fineweb_data

def main():
    # datasets_dir = '/localhome/imodoran/datasets/'
    datasets_dir = '/mnt/beegfs/alistgrp/imodoran/datasets/llm-baselines/'

    local_folder = "fineweb-edu"
    hf_path = f"HuggingFaceFW/{local_folder}"
    subset = "sample-10BT"
    get_fineweb_data(datasets_dir, hf_path, subset, local_folder, num_proc=64)


if __name__ == '__main__':
    main()
