# Codebase for [DASH: Faster Shampoo via Batched Block Preconditioning and Efficient Inverse-Root Solvers](https://arxiv.org/pdf/2602.02016)

This repository contains the code for DASH (**D**istributed **A**ccelerated **SH**ampoo), a faster implementation of DistributedShampoo 
optimizer. We summarize our contribution as follows:

- Our key observation is that DistributedShampoo processes the preconditioner blocks sequentially, which make it slow. In **DASH**, we
  stack the blocks with the same size in a 3D tensor and process them in parallel to increase the GPU utilization.
- We investigate two new lienar algebraic approaches to compute matrix powers: **Newton-DB** and **Chebyshev polynomials via Clenshaw's 
  algorithm**. **NewtonDB** achieves lowest validation perplexity, better than even **Eigen-Value Decomposition (EVD)**.
- We probide a behavioral analysis of Newton-based iterations and show that scaling the matrices using Frobenius norm is suboptimal, 
  thus requiring more iterations to reach the desired precision. We offer intuitive explanations between Coupled Newton and Newton-DB.
- Our **DASH** implementation enables running Power-Iteration in an efficient way by estimating the largest eigenvectors in parallel to 
  maximize GPU efficiency. On top of the parallelization introduced by stacking the matrices, we run Power-Iteration on multiple vectors 
  at the same time to avoid getting stuck in a local maxima (we call this **Multi-Power-Iteration**).

## **DASH** implementation
Our implementation can be found in the repository [ISTA-DASLab-Optimizers](https://github.com/IST-DASLab/ISTA-DASLab-Optimizers). If you 
follow the instructions in the **Reproducing Experiments**, the **DASH** implementation will be available via the 
`ista-daslab-optimizers`, which is part of [requirements.txt](https://github.com/IST-DASLab/DASH/blob/main/requirements.txt#L5) file. 

## Reproducing Experiments
We use the C4 dataset from [ISTA-DASLab/C4-tokenized-llama2](https://huggingface.co/datasets/ISTA-DASLab/C4-tokenized-llama2) that is 
already tokenized. You can use [this script](https://github.com/IST-DASLab/DASH/blob/main/src/data/hf_hub_download.py) to download the 
chunks of the tokenized C4 dataset from HuggingFace. 

This repository is based on [llm-baselines](https://github.com/epfml/llm-baselines) from EPFL.

We use [GridSearcher](https://github.com/IST-DASLab/GridSearcher) to efficiently run multiple grid-search configurations on a compute 
node where we had access to 8x H100 GPUs. **GridSearcher** is automatically installed via the
[requirements.txt](https://github.com/IST-DASLab/DASH/blob/main/requirements.txt#L13) file.

```bash
#!/bin/bash

ROOT=/tmp/ISTA-DASLab # change it according to your needs

mkdir -p $ROOT
mkdir -p ${ROOT}/datasets/c4
mkdir -p ${ROOT}/results

cd $ROOT

######################### CREATE & ACTIVATE "DASH" ENVIRONMENT
conda create --name DASH python=3.12 -c conda-forge --override-channels -y
conda activate DASH

######################### CLONE DASH (THIS REPO)
git clone git@github.com:IST-DASLab/DASH.git
cd $ROOT/DASH

######################### REQUIREMENTS & LOGINs
pip install -r requirements.txt
# make sure you are logged to HuggingFace and WandB (we skip the login part, please do that yourself!)

######################### INSTALL OUR FORK OF DISTRIBUTED SHAMPOO WITH CHANGES INTRODUCED IN DASH
cd $ROOT
git clone git@github.com:IST-DASLab/DASH_DistributedShampoo.git
cd DASH_DistributedShampoo
pip install -e .

######################### DOWNLOAD THE TOKENIZED C4 DATASET
HF_PATH_C4=ISTA-DASLab/C4-tokenized-llama2
LOCAL_PATH_C4=/tmp/ISTA-DASLab/datasets/c4

python3 ${ROOT}/DASH/src/data/hf_hub_download.py --repo_id=$HF_PATH_C4 --local_dir=$LOCAL_PATH_C4

cd $LOCAL_PATH_C4
cat chunk_* > train.bin
# now we can use train.bin and val.bin from $LOCAL_PATH_C4

######################### RUN DASH WITH GridSearcher (REQUIRES DIRECT SSH ACCESS TO THE MACHINE WITH 8 GPUs)
cd $ROOT/DASH/src
python3 run_dash.py --wandb_entity=ionutmodo # replace with your desired WandB entity
```

## Alternative to GridSearcher
If you do not want to run experiments with GridSearcher, then you can use it just to create the bash commands as a result of the 
cartesian product of your grid runs by setting the flag `debug=True` when calling `gs.run(...)` [here](https://github.com/IST-DASLab/DASH/blob/main/src/run_dash.py#L267).

## Issues 🤝
Please open an issue if you have questions, observations or find bugs in our code! We are open to discussions and we will try to address 
them as soon as possible! Thank you in advance! 

## Roadmap:
- ⏳ Working on an even more efficient version (work in progress)
- ⏳ Release model checkpoints on HuggingFace (work in progress)
- ✅ Release our fork of Distributed Shampoo (done on 2026-feb-06)
- ✅ Code Release: DASH (this repo) and in [ISTA-DASLab-Optimizers](https://github.com/IST-DASLab/ISTA-DASLab-Optimizers) (done on 
  2026-feb-06)
- ✅ Upload Paper on Arxiv (done on 2026-feb-02)

## Citation
If you find our work useful, please consider citing:
```
@misc{modoranu2026dash,
      title={DASH: Faster Shampoo via Batched Block Preconditioning and Efficient Inverse-Root Solvers}, 
      author={Ionut-Vlad Modoranu and Philip Zmushko and Erik Schultheis and Mher Safaryan and Dan Alistarh},
      year={2026},
      eprint={2602.02016},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.02016}, 
}
```