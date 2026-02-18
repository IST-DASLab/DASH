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

### Available versions:
1. **DashLayerwise**: this version performs stacking at each layer level and it was used to generate the results in
the paper  
2. **DashGpu**: this version was developed after releasing the paper and it stacks all blocks from all layers
allocated to one GPU, which results in slightly faster running time and slightly higher memory consumption because
the tensors we allocate in this version are larger.

**Block size 2048:**

| Root Method  | Frequency | Time for Distributed Shampoo | Time for DASH-Layerwise | Time for DASH-GPU | NORM / PREC |
|--------------|-----------|------------------------------|-------------------------|-------------------|-------------|
| EVD          | 1         | 2200 ms                      | 1747 ms                 | 1755 ms           | -           |
| EVD          | 10        | 253 ms                       | 209 ms                  | 210 ms            | -           |
| CN           | 1         | 675 ms                       | 221 ms                  | **207 ms**        | FRO / FP32  |
| CN           | 1         | 243 ms                       | 169 ms                  | **153 ms**        | FRO / FP16  |
| NDB          | 1         | x ms                         | 279 ms                  | **264 ms**        | FRO / FP32  |
| NDB          | 1         | 355 ms                       | 284 ms                  | **267 ms**        | PI / FP32   |
 
**Block size 1024:**

| Root Method  | Frequency | Time for Distributed Shampoo | Time for DASH-Layerwise | Time for DASH-GPU | NORM / PREC |
|--------------|-----------|------------------------------|-------------------------|-------------------|-------------|
| EVD          | 1         | 3080 ms                      | 2850 ms                 | 2850 ms           | -           |
| EVD          | 10        | 355 ms                       | 315 ms                  | 313 ms            | -           |
| CN           | 1         | 666 ms                       | 149 ms                  | **136 ms**           | FRO / FP32  |
| CN           | 1         | 471 ms                       | 138 ms                  | **119 ms**           | FRO / FP16  |
| NDB          | 1         | 558 ms                       | 188 ms                  | **174 ms**           | FRO / FP32  |
| NDB          | 1         | 740 ms                       | 194 ms                  | **177 ms**           | PI / FP32   |

 Comparing the running time of Distributed Shampoo for **CN-1-FP32** and *DASH-GPU* for **CN-1-FP16**, we get a reduction of `666ms / 
 119ms = 5.6x`, compared to `4.83x` reported in the paper for *DASH-Layerwise*.

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