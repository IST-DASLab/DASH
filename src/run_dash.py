import os
import sys
sys.path.append(os.getcwd())

from string import Template
import time
import os
from gridsearcher import GridSearcher, SchedulingConfig, TorchRunConfig
from tqdm import tqdm
import argparse
import psutil
import math
import socket
import torch
import gpustat

def wait_for_pids(pids):
    while any([psutil.pid_exists(int(pid)) for pid in pids]):
        print(f'waiting for processes {pids} to end...')
        time.sleep(60)

def get_pids_per_gpu():
    num_gpus = 8
    pids = [0] * num_gpus
    gpus = gpustat.new_query().gpus
    breakpoint()
    for gid in range(num_gpus):
        for p in gpus[gid]['processes']:
            pids[gid] = p['pid']
    return pids

def get_arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait_pids', nargs='+', default=None, required=False)
    parser.add_argument('--wait_secs', type=int, default=None, required=False)
    parser.add_argument('--wait_current', action="store_true", default=False)
    parser.add_argument('--wandb_entity', type=str, required=True)
    return parser.parse_args()

def add_model_params(model_size, tokens_ratio, seq_len, batch_size, acc_steps):
    data = {}
    if model_size == 3:
        data['n_layer'] = 1
        data['n_embd'] = 512
        data['n_head'] = 4
    elif model_size == 30:
        # tokens = 3_000_000_000
        # data['lr'] = 0.0012
        data['n_layer'] = 6
        data['n_embd'] = 640
        data['n_head'] = 5
    elif model_size == 50:
        # tokens = 5_000_000_000
        # data['lr'] = 0.0012
        data['n_layer'] = 7
        data['n_embd'] = 768
        data['n_head'] = 6
    elif model_size == 100:
        # tokens = 10_000_000_000
        # data['lr'] = 0.0006
        data['n_layer'] = 8
        data['n_embd'] = 1024
        data['n_head'] = 8
    elif model_size == 200:
        # tokens = 20_000_000_000
        # data['lr'] = 0.0003
        data['n_layer'] = 10
        data['n_embd'] = 1280
        data['n_head'] = 10
    elif model_size == 350:  # from Dion
        # data['lr'] = 0.01
        data['n_layer'] = 24
        data['n_embd'] = 1024
        data['n_head'] = 32
    elif model_size == 360:  # from SOAP
        # data['lr'] = 0.01
        data['n_layer'] = 24
        data['n_embd'] = 1024
        data['n_head'] = 64
    elif model_size == 430:
        # tokens = 43_000_000_000
        # data['lr'] = 0.00015
        data['n_layer'] = 13
        data['n_embd'] = 1664
        data['n_head'] = 13
    elif model_size == 800:
        # tokens = 80_000_000_000
        # data['lr'] = 0.000075
        data['n_layer'] = 16
        data['n_embd'] = 2048
        data['n_head'] = 16
    elif model_size == 869:
        # data['lr'] = 0.000075
        data['n_layer'] = 3
        data['n_embd'] = 4096
        data['n_head'] = 16
    elif model_size == 953:
        # tokens = 80_000_000_000
        # data['lr'] = 0.000075
        data['n_layer'] = 16
        data['n_embd'] = 2048
        data['n_head'] = 16
    elif model_size == 1300: # from Dion
        # data['lr'] = 0.01
        data['n_layer'] = 24
        data['n_embd'] = 2048
        data['n_head'] = 32
    elif model_size == 1700:
        # tokens = 10_750_000_000
        data['n_layer'] = 20
        data['n_embd'] = 2688
        data['n_head'] = 21
    elif model_size == 3200:
        # tokens = 20_000_000_000
        # data['lr'] = 0.000075
        data['n_layer'] = 28
        data['n_embd'] = 3072
        data['n_head'] = 24
    elif model_size == 6700:
        # data['lr'] = 0.000075
        data['n_layer'] = 32 # 6.7B for 32 layers, 8.3B for 40 layers
        data['n_embd'] = 4096
        data['n_head'] = 32

    tokens = model_size * 1_000_000 * tokens_ratio
    iterations = math.ceil(tokens / (batch_size * acc_steps * seq_len))

    data['warmup_steps'] = math.ceil(iterations / 10) # 10%
    data['iterations'] = iterations

    return data

def main(MODEL_SIZE, OPTIM, ROOT_METHOD, TOKENS_RATIO, SEQ_LEN, BATCH_SIZE, ACC_STEPS,
         gpus,
         wandb_project,
         group_prefix='', dist=False, max_jobs=1, grid_dict=None, add_algo=False):
    script_path = './main.py'
    datasets_dir = '/tmp/ISTA-DASLab/datasets/'
    exp_folder_root = f'/tmp/ISTA-DASLab/results/'

    gs = GridSearcher(script=script_path, defaults=dict())

    gs.add_param('latest_ckpt_interval', 1000)
    gs.add_param('compile', True)
    gs.add_param('model', 'llama')

    gs.add_param('vocab_size', 32_000)

    gs.add_param('batch_size', BATCH_SIZE)
    gs.add_param('acc_steps', ACC_STEPS)

    # gs.add_param('dataset', 'c4')
    gs.add_param('datasets_dir', datasets_dir)
    # gs.add_param('datasets_dir', '/dev/shm/')

    data = add_model_params(model_size=MODEL_SIZE,
                            tokens_ratio=TOKENS_RATIO,
                            seq_len=SEQ_LEN,
                            batch_size=BATCH_SIZE,
                            acc_steps=ACC_STEPS)
    for k, v in data.items(): # sets values for n_layer, n_embd, n_head, lr, iterations, warmup_steps
        gs.add_param(k, v)

    gs.add_param('opt', OPTIM)
    gs.add_param('wandb', True)

    wandb_job_type = 'lr=${lr}'

    if OPTIM == 'adamw':
        # wandb_group = '${opt}_b=${adamw_beta1}-${adamw_beta2}_eps=${adamw_eps}_wd=${weight_decay}'
        wandb_group = '${opt}_wd=${weight_decay}'
    elif OPTIM == 'dist-shmp':
        if ROOT_METHOD == 'evd':
            wandb_group = ('${opt}_'
                           '${shmp_inv_root_method}_'
                           'f=${shmp_inv_root_freq}_'
                           'eps=${shmp_eps_inv_root}_'
                           'mpd=${shmp_max_prec_dim}')
        elif ROOT_METHOD == 'cn':
            wandb_group = ('${opt}_'
                           '${shmp_inv_root_method}_'
                           'f=${shmp_inv_root_freq}_'
                           'it=${shmp_newton_steps}_'
                           'eps=${shmp_eps_inv_root}_'
                           'dtype=${shmp_matmul_dtype}_'
                           'mpd=${shmp_max_prec_dim}_'
                           'sc=${shmp_matrix_scaling_type}-${shmp_matrix_scaling_pi_steps}-${shmp_matrix_scaling_const}')
        elif ROOT_METHOD == 'ndb':
            wandb_group = ('${opt}_'
                           '${shmp_inv_root_method}_'
                           'f=${shmp_inv_root_freq}_'
                           'it=${shmp_newton_steps}_'
                           'dtype=${shmp_matmul_dtype}_'
                           'mpd=${shmp_max_prec_dim}_'
                           'sc=${shmp_matrix_scaling_type}-${shmp_matrix_scaling_pi_steps}-${shmp_matrix_scaling_const}')
        elif ROOT_METHOD == 'cbshv':
            wandb_group = ('${opt}_'
                           '${shmp_inv_root_method}-v${shmp_clenshaw_version}_'
                           'f=${shmp_inv_root_freq}_'
                           'd=${shmp_cbshv_degree}_'
                           'dtype=${shmp_matmul_dtype}_'
                           'mpd=${shmp_max_prec_dim}')
    elif OPTIM in ['dash-lw']:
        if ROOT_METHOD == 'evd':
            wandb_group = ('${opt}_'
                           '${shmp_inv_root_method}_'
                           'f=${shmp_inv_root_freq}_'
                           'eps=${shmp_eps_inv_root}_'
                           'mpd=${shmp_max_prec_dim}_'
                           'evdh=${shmp_evd_heuristic}')
        elif ROOT_METHOD == 'cn':
            wandb_group = ('${opt}_'
                           '${shmp_inv_root_method}_'
                           'f=${shmp_inv_root_freq}_'
                           'it=${shmp_newton_steps}_'
                           'dtype=${shmp_matmul_dtype}_'
                           'mpd=${shmp_max_prec_dim}_'
                           'sc=${shmp_matrix_scaling_type}-${shmp_matrix_scaling_pi_steps}-${shmp_matrix_scaling_const}')
        elif ROOT_METHOD == 'ndb':
            wandb_group = ('${opt}_'
                           '${shmp_inv_root_method}_'
                           'f=${shmp_inv_root_freq}_'
                           'it=${shmp_newton_steps}_'
                           'dtype=${shmp_matmul_dtype}_'
                           'mpd=${shmp_max_prec_dim}_'
                           'sc=${shmp_matrix_scaling_type}-${shmp_matrix_scaling_pi_steps}-${shmp_matrix_scaling_const}')
        elif ROOT_METHOD == 'cbshv':
            wandb_group = ('${opt}_'
                           '${shmp_inv_root_method}_'
                           'f=${shmp_inv_root_freq}_'
                           'd=${shmp_cbshv_degree}_'
                           'dtype=${shmp_matmul_dtype}_'
                           'mpd=${shmp_max_prec_dim}_'
                           'sc=${shmp_matrix_scaling_type}-${shmp_matrix_scaling_pi_steps}-${shmp_matrix_scaling_const}')

        if add_algo:
            wandb_group += '_algo1d=${shmp_algo_one_dim}'
    else:
        raise RuntimeError(f'Optimizer {OPTIM} is currently not supported')

    if len(group_prefix) > 0: wandb_group = group_prefix + '_' + wandb_group
    wandb_run_prefix = f'{MODEL_SIZE}M-tpp={TOKENS_RATIO}'
    gs.add_param('wandb_run_prefix', wandb_run_prefix)
    gs.add_param('wandb_group', Template(wandb_group))
    gs.add_param('wandb_job_type', Template(wandb_job_type))
    gs.add_param('wandb_project', wandb_project)

    if dist:
        gs.add_param('distributed_backend', 'nccl')

    gs.run(
        param_name_for_exp_root_folder='results_base_folder',
        exp_folder=Template(exp_folder_root + f'{gs._wandb_project}/{wandb_run_prefix}_{wandb_group}_{wandb_job_type}' + '_dataseed=${data_seed}'),
        cfg_sched=SchedulingConfig(
            distributed_training=dist,
            max_jobs_per_gpu=max_jobs,
            gpus=gpus,
            params_values=grid_dict,
        ),
        cfg_torchrun=TorchRunConfig(
            launch_blocking=0,
            torchrun=dist,
            master_addr='127.0.0.1',
            master_port=29500,
            rdzv_backend='static' if 'gpu413' in socket.gethostname() else 'c10d',
        ),
        # debug=True,
    )

if __name__ == '__main__':
    args = get_arg_parse()

    if args.wait_current: # if this option is set, then we will wait for all GPUs to be free (meaning no more processes running on GPUs)
        wait_for_pids(pids=get_pids_per_gpu())
    else:
        if args.wait_pids is not None: # alternatively, we can wait for explicit PIDs (it can also be the PID of an already running GridSearcher script)
            wait_for_pids(pids=args.wait_pids)

        if args.wait_secs is not None: # alternatively, we can wait a specific amount of seconds before we start the script
            print(f'Waiting {args.wait_secs} seconds')
            for _ in tqdm(range(args.wait_secs)):
                time.sleep(1)

    ######################################################################################################################################################
    ######################################################################################################################################################
    ######################################################################################################################################################

    SEQ_LEN = 1024
    dataset = 'c4'

    dist = True ########## will add backend=nccl in main() above
    # dist = False

    WANDB_GROUP_PREFIX = '' ########## if you want to identify the run by prefix

    ADD_ALGO_ONE_DIM_TO_WANDB_GROUP = True ########## keep to True

    ########## here we start defining the values that will be used in the grid with for loops
    for ROOT_METHOD in [ ########## choose inverse root method from here (or all)
        'evd',
        # 'ndb',
        # 'cn',
        # 'cbshv',
    ]:
        is_cn = ROOT_METHOD == 'cn'
        is_ndb = ROOT_METHOD == 'ndb'
        is_evd = ROOT_METHOD == 'evd'
        is_cbshv = ROOT_METHOD == 'cbshv'

        for MODEL_SIZE, MAX_PREC_DIM, TOKENS_RATIO, DEVICE_BATCH_SIZE, ACC_STEPS in [ ########## choose model size from here (or all)
            ########## MODELS FOR DEBUGGING
            # (  3,  512, 20, 64,   2), # GlobalBatchSize = 131k: extremely small model for debugging
            # ( 30,  640, 20, 64,  32), # GlobalBatchSize =   2M: small model to check validation loss quickly

            ########## SMALL SCALE MODELS
            # (360, 1024, 20, 128,  16), # GlobalBatchSize = 2M: DEVICE_BATCH_SIZE=128 for B200-180GB
            # (360, 1024, 20,   8, 256), # GlobalBatchSize = 2M: DEVICE_BATCH_SIZE=128 for B200-180GB

            ########## USED IN DASH PAPER
            (953, 1024, 20, 32, 64), # GlobalBatchSize = 2M for H100-80GB
            # (953, 2048, 20, 32, 64), # GlobalBatchSize = 2M for H100-80GB
        ]:
            for OPTIM, GPUS in [ ########## choose optimizer and GPUs
                ('dash-lw', [0, 1, 2, 3, 4, 5, 6, 7]), ########## DashLayerwise
                ('dist-shmp', [0, 1, 2, 3, 4, 5, 6, 7]), ########## DistributedShampoo (our modified version)
            ]:
                ########## some stats for each model
                eval_interval = {3: 10, 30: 10, 360: 50, 953: 100}[MODEL_SIZE] ########## how often to compute validation loss
                log_interval = 1 ########## how often to upload logs to wandb (keep it to 1)
                stat_interval = {3: 1, 30: 1, 360: 50, 953: 500}[MODEL_SIZE] ########## how often to compute layer statistics in DASH

                grid = { ########## key=argument-value(from config.py), value=list-of-values-for-key
                    'eval_interval': [eval_interval],
                    'log_interval': [log_interval],
                    'shmp_log_stats_interval': [stat_interval],
                    'dataset': [dataset],
                    'sequence_length': [SEQ_LEN],
                    'wandb_entity': [
                        args.wandb_entity
                    ],
                    'data_seed': [########## change the seeds for reproducibility
                        42,
                        # 666,
                        # 2408,
                    ],
                    'lr': [ ########## we used lr=1e-3
                        '1e-3',
                        # '2e-3',
                        # '3e-3',
                    ],
                    ##################################################
                    # 'limit_iters_to': [ ########## uncomment this to run only 200 iterations to check timings or other stats
                    #     200,
                    # ],
                    ##################################################
                    'weight_decay': ['0'],
                    'shmp_max_prec_dim': [MAX_PREC_DIM],

                    ### DASH
                    'shmp_algo_one_dim': [ ########## how to update normalization layers: adam or shmp (adagrad)
                        # 'adamw',
                        'shmp',
                    ],
                    'shmp_eps_inv_root': [ ########## regularization
                        # '0',
                        '1e-10',
                    ],
                    'shmp_inv_root_freq': [ ########## this is the FREQ from Table 1 (how often to compute the inverse roots)
                        1,
                        # 10,
                    ],
                    'shmp_matrix_scaling_type': [ ########## matrix scaling type
                        # 'pi', # Power-Iteration
                        'fro', # Frobenius norm
                        # 'pim', # Power-Iteration-Multi
                    ],
                    'shmp_matmul_dtype': [ ########## desired precision (see CHECKS section from function main() in main.py)
                        'fp32',
                        # 'fp16',
                        # 'bf16',
                    ],

                    ### DASH: EVD
                    'shmp_evd_heuristic': [ ########## we used "shmp" for paper to match DistributedShampoo, the others are experimental
                        # 'abs',
                        # 'abs-add',
                        # 'relu',
                        'shmp',
                    ],

                    ### DASH: NewtonDB & CoupledNewton
                    'shmp_newton_steps': [ ########## number of iterations for NewtonDB and CoupledNewton
                        10,
                    ],

                    ### DASH: Chebyshev
                    'shmp_cbshv_degree': [ ########## degree for Chebyshev
                        # 40,
                        60,
                        # 80,
                        # 100,
                    ],

                    ### DASH: CoupledNewton
                    'shmp_cn_tolerance': ['1e-6'], ########## the default desired tolerance for CoupledNewton in DistributedShampoo (do not change)

                    ### fixed settings
                    'shmp_inv_root_method': [ROOT_METHOD],
                    'shmp_matrix_scaling_pi_steps': [10], ########## 10 steps for Power-Iteration
                    'shmp_matrix_scaling_const': [2], ########## multiply the approximation of largest eigenvalue by 2
                    'shmp_grafting_type': ['adam'], ########## we only implement Adam
                    'shmp_grafting_eps': ['1e-8'], ########## keep fixed

                    'shmp_mu': ['0'], ########## no momentum: if you set this to 0.9, both DIST and DASH will diverge
                    'shmp_use_nesterov': ['0'], ########## used only when shmp_mu > 0
                    'shmp_start_prec_step': [-1], ########## compute inverses when current_iter >= shmp_start_prec_step (keep to -1)
                    'shmp_beta_g': ['0.9'], ########## kept fixed in DASH paper
                    'shmp_beta_lr': ['0.95'], ########## kept fixed in DASH paper
                    'shmp_beta_graft': ['0.95'], ########## kept fixed in DASH paper

                    'adamw_eps': ['1e-8'], ########## used only when we update Normalization Layers with Adam (kept fixed in DASH paper)
                    'adamw_beta1': ['0.9'], ########## used only when we update Normalization Layers with Adam (kept fixed in DASH paper)
                    'adamw_beta2': ['0.95'], ########## used only when we update Normalization Layers with Adam (kept fixed in DASH paper)

                    # DistributedShampoo
                    'shmp_beta1': ['0.9'], ########## kept fixed in DASH paper
                    'shmp_beta2': ['0.95'],  ########## kept fixed in DASH paper
                    'shmp_clenshaw_version': [51], ########## kept fixed in DASH paper
                    'shmp_beta_prec': ['0.95'], ########## kept fixed in DASH paper
                    'shmp_beta_gg': ['0.05'], ########## kept fixed in DASH paper
                }

                ########## This function runs GridSearcher using the parameters grid in "params"
                ########## All parameters in the grid will be employed in the templates of GridSearcher
                ########## to embed their values in disk paths automatically
                main(
                    MODEL_SIZE=MODEL_SIZE,
                    OPTIM=OPTIM,
                    ROOT_METHOD=ROOT_METHOD,
                    TOKENS_RATIO=TOKENS_RATIO,
                    SEQ_LEN=SEQ_LEN,
                    BATCH_SIZE=DEVICE_BATCH_SIZE,
                    ACC_STEPS=ACC_STEPS,
                    wandb_project=f'DASH',
                    group_prefix=WANDB_GROUP_PREFIX,
                    gpus=GPUS,
                    dist=dist,
                    grid_dict=grid,
                    max_jobs=1,
                    add_algo=ADD_ALGO_ONE_DIM_TO_WANDB_GROUP
                )
