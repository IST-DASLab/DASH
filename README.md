# Codebase for [DASH: Faster Shampoo via Batched Block Preconditioning and Efficient Inverse-Root Solvers](https://arxiv.org/pdf/2602.02016)

## Quickstart

We use the C4 dataset from [ISTA-DASLab/C4-tokenized-llama2](https://huggingface.co/datasets/ISTA-DASLab/C4-tokenized-llama2) that is 
already tokenized. You can find a script

```bash
#!/bin/bash

ROOT=/tmp/ISTA-DASLab # change it according to your needs

mkdir -p $ROOT
mkdir -p ${ROOT}/datasets/c4
mkdir -p ${ROOT}/results

cd $ROOT

##### CREATE & ACTIVATE "DASH" ENVIRONMENT
conda create --name DASH python=3.12 -c conda-forge --override-channels -y
conda activate DASH
#########################

pip install huggingface-hub

# we skip the login part to HuggingFace (please do that yourself)

##### CLONE DASH (THIS REPO)
git clone git@github.com:IST-DASLab/DASH.git
cd $ROOT/DASH
#########################

pip install -r requirements.txt

##### INSTALL DISTRIBUTED SHAMPOO WITH CHANGES INTRODUCED IN DASH
cd $ROOT
git clone git@github.com:IST-DASLab/distributed_shampoo.git
cd distributed_shampoo
pip install -e .
######################### END INSTALL DISTRIBUTED SHAMPOO

##### DOWNLOAD C4 DATASET
HF_PATH_C4=ISTA-DASLab/C4-tokenized-llama2
LOCAL_PATH_C4=/tmp/ISTA-DASLab/datasets/c4

python3 ${ROOT}/DASH/src/data/hf_hub_download.py --repo_id=$HF_PATH_C4 --local_dir=$LOCAL_PATH_C4

cd $LOCAL_PATH_C4
cat chunk_* > train.bin
######################### END DOWNLOAD C4 DATASET: Now we can use train.bin and val.bin from $LOCAL_PATH_C4

##### RUN DASH WITH GridSearcher (REQUIRES DIRECT SSH ACCESS TO THE MACHINE WITH 8 GPUs)
cd $ROOT/DASH/src
python3 run_dash.py
```

### Alternative to GridSearcher
If you do not want to run experiments with GridSearcher, then you can only instruct it to return the bash command it runs by setting the
flag `debug=True` when calling `gs.run(...)`. 
