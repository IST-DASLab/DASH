import warnings; warnings.filterwarnings("ignore")
import os
import sys
import json
import torch
import wandb
import random
import socket
import argparse
import numpy as np
from pathlib import Path
import torch.distributed as dist

from optim.utils import cos_inf_schedule, wsd_schedule, load_wandb_state
from data.utils import DataReader, get_dataset
from models.utils import get_model
from optim.optim import train
import distributed
import config

from ista_daslab_optimizers import * # imports all DashXyz names

from distributed_shampoo import (
    AdamPreconditionerConfig,
    DistributedShampoo,
    DDPDistributedConfig,
    SingleDeviceDistributedConfig,
    ShampooPT2CompileConfig,
    RootInvShampooPreconditionerConfig,
    CoupledNewtonConfig,
    NewtonSchulzInvRootConfig,
    ChebyshevInvRootConfig,
    NewtonDBInvRootConfig,
    EigenConfig,
)

FP32 = torch.float32
FP16 = torch.float16
BF16 = torch.bfloat16

def main(args):
    distributed_backend = distributed.make_backend_from_args(args)
    args = distributed_backend.get_adjusted_args_for_process(args)
    args.world_size = distributed_backend.get_world_size()
    args.devices_count = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

    ####################
    ##### CHECKS
    ##### Exit when the following invalid settings are detected
    ##### This helps with more efficient parallelization with GridSearcher
    ##### Examples of invalid setting:
    #####      opt=dist-shmp, shmp_inv_root_method=cn, shmp_matrix_scaling_type=pi
    #####      Explanation: DistributedShampoo supports only scaling by frobenius norm (no PowerIteration)
    if args.opt == 'dist-shmp':
        if args.shmp_inv_root_method == 'evd' and args.shmp_matmul_dtype in ['fp16', 'bf16']:
            sys.exit(666)

        if args.shmp_inv_root_method == 'cn' and args.shmp_matrix_scaling_type == 'pi':
            sys.exit(666)

    if ('clr-adamw' in args.opt) and (args.lowrank_use_ef == 0) and (args.lowrank_q_ef != 0):
        sys.exit(666)
    ##### END CHECKS
    ####################

    if args.full_eval_at is None:
        args.full_eval_at = []

    # NOTE args.seed is offset per worker in get_adjusted_args_for_process
    # torch.use_deterministic_algorithms(True)  # CUBLAS_WORKSPACE_CONFIG=:4096:8
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if "cuda" in args.device:
        torch.cuda.set_device(torch.device(args.device))

    exp_name = get_exp_name(args, distributed_backend)
    exp_dir = Path(args.results_base_folder) / exp_name

    print(f"Starting Experiment: {exp_name}")
    print(f"Experiment Directory: {exp_dir}")
    print(f"Config:\n{vars(args)}\n")

    print(f"Loading dataset: '{args.dataset}'")
    datareaders = get_data_readers(args)

    model = get_model(args).to(device=args.device)

    # TODO: take care of initializing the model if args.use_pretrained != 'none'
    print(f"\nModel:\n{model}")

    model = distributed_backend.transform_model(model)
    group_specs = distributed_backend.get_raw_model(model).get_parameter_group_specs()
    param_name_mapping = {p_name: p for p_name, p in model.named_parameters()}
    optimized_params_cnt = 0
    for g in group_specs:
        params = []
        for p_name in g["params"]:
            translated_p_names = (
                distributed_backend.translate_model_parameter_name_for_node(p_name)
            )
            params += [param_name_mapping[p_name] for p_name in translated_p_names]
        g["params"] = params
        optimized_params_cnt += sum([p.numel() for p in g["params"]])
    params_cnt = distributed_backend.get_raw_model(model).get_num_params()
    nonemb_param_cnt = (
        params_cnt
        - distributed_backend.get_raw_model(model).lm_head.weight.numel()
        - distributed_backend.get_raw_model(model).transformer.wte.weight.numel()
    )
    print("number of parameters: %.2fM" % (params_cnt / 1e6,))
    print("number of optimized parameters: %.2fM" % (optimized_params_cnt / 1e6,))
    print("number of non-embedding parameters: %.2fM" % (nonemb_param_cnt / 1e6,))

    if args.opt == "adamw":
        opt = torch.optim.AdamW(
            group_specs,
            lr=args.lr,
            betas=(args.adamw_beta1, args.adamw_beta2),
            weight_decay=args.weight_decay,
            eps=args.adamw_eps,
        )
    elif args.opt == 'dist-shmp':
        matmul_dtype = {
            'fp16': torch.float16,
            'bf16': torch.bfloat16,
            'fp32': torch.float32,
        }[args.shmp_matmul_dtype]
        match args.shmp_inv_root_method:
            case 'evd':
                amortized_cfg = EigenConfig()
            case 'cn':
                amortized_cfg = CoupledNewtonConfig(dtype=matmul_dtype, max_iterations=args.shmp_newton_steps, tolerance=args.shmp_cn_tolerance)
            case 'ns':
                amortized_cfg = NewtonSchulzInvRootConfig()
            case 'ndb':
                amortized_cfg = NewtonDBInvRootConfig(
                    steps=args.shmp_newton_steps,
                    root4_src=args.shmp_ndb_root4_src,
                    matmul_dtype=matmul_dtype,
                    scaling=args.shmp_matrix_scaling_type,
                    pi_iters=args.shmp_matrix_scaling_pi_steps,
                    pi_const=args.shmp_matrix_scaling_const,
                )
            case 'cbshv':
                amortized_cfg = ChebyshevInvRootConfig(
                    degree=args.shmp_cbshv_degree,
                    matmul_dtype=matmul_dtype,
                    clenshaw_version=args.shmp_clenshaw_version,
                )
            case _:
                raise RuntimeError(f'Invalid method: {args.shmp_inv_root_method}')

        opt = DistributedShampoo(
            model.parameters(),
            lr=args.lr,
            epsilon=args.shmp_eps_inv_root,
            weight_decay=args.weight_decay,
            betas=(args.shmp_beta1, args.shmp_beta2),
            beta_prec=args.shmp_beta_prec,
            beta_gg=args.shmp_beta_gg,
            momentum=args.shmp_mu,
            precondition_frequency=args.shmp_inv_root_freq,
            use_nesterov=bool(args.shmp_use_nesterov),

            use_decoupled_weight_decay=True,
            max_preconditioner_dim=args.shmp_max_prec_dim, # default 1024
            start_preconditioning_step=args.shmp_start_prec_step,
            use_bias_correction=True,
            grafting_config=AdamPreconditionerConfig(beta2=args.shmp_beta2, epsilon=args.adamw_eps),
            distributed_config=SingleDeviceDistributedConfig() if args.devices_count == 1 else DDPDistributedConfig(),
            preconditioner_config=RootInvShampooPreconditionerConfig(amortized_computation_config=amortized_cfg),
            # shampoo_pt2_compile_config=ShampooPT2CompileConfig(),
        )
    elif args.opt in ['dash-lw', 'dash-gpu']:
        dash_ctor = DashLayerwise if args.opt == 'dash-lw' else DashGpu
        opt = dash_ctor(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            config=DashConfig(
                adamw_eps=args.adamw_eps,
                adamw_beta1=args.adamw_beta1,
                adamw_beta2=args.adamw_beta2,

                beta_G=args.shmp_beta_g,
                beta_LR=args.shmp_beta_lr,
                beta_graft=args.shmp_beta_graft,

                eps_inv_root=args.shmp_eps_inv_root,
                inv_root_method=DashInverseRootMethodType.from_string(args.shmp_inv_root_method),
                inv_root_freq=args.shmp_inv_root_freq,

                grafting_type=DashGraftingType.from_string(args.shmp_grafting_type),
                eps_grafting=args.shmp_grafting_eps,

                mu=args.shmp_mu,
                use_nesterov=bool(args.shmp_use_nesterov),
                use_bias_correction=True, # no corresponding --use_bas_correction arg in config for this

                start_prec_step=args.shmp_start_prec_step,
                block_size=args.shmp_max_prec_dim,
                matmul_dtype={'fp32': FP32, 'fp16': FP16, 'bf16': BF16}[args.shmp_matmul_dtype],

                matrix_scaling_type=DashMatrixScalingType.from_string(args.shmp_matrix_scaling_type),
                matrix_scaling_pi_steps=args.shmp_matrix_scaling_pi_steps,
                matrix_scaling_const=args.shmp_matrix_scaling_const,

                newton_steps = args.shmp_newton_steps, # for NewtonDB and CoupledNewton
                algo_one_dim=DashAlgoOneDim.from_string(args.shmp_algo_one_dim),

                ### EVD
                evd_heuristic=DashEVDHeuristic.from_string(args.shmp_evd_heuristic),

                ### CN
                cn_tolerance=args.shmp_cn_tolerance,

                ### CBSHV
                cbshv_degree=args.shmp_cbshv_degree,
            )
        )
    else:
        raise RuntimeError(f"Unknown opt: {args.opt}")
    print(f"\nOptimizer:\n{opt}")

    if args.scheduler != "none":
        assert args.warmup_steps < args.iterations, "Warmup steps must be < iterations."
        if args.scheduler in ["cos", "linear"]:
            # initial lr is args.lr / div_factor
            # final lr is initial_lr/final_div_factor = args.lr / div_factor / final_div_factor
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=opt,
                max_lr=[args.lr] if args.opt in ['dist-shmp', 'dash-lw', 'dash-gpu'] else [group.get("lr", args.lr) for group in group_specs],
                total_steps=args.iterations,
                pct_start=args.warmup_steps / args.iterations,
                anneal_strategy=args.scheduler,
                cycle_momentum=False,
                div_factor=1e2,
                final_div_factor=0.1,
            )
        elif args.scheduler == "cos_inf":
            lambda_schedule = cos_inf_schedule(
                n_iterations=args.iterations,
                n_warmup=args.warmup_steps,
                n_inf=args.cos_inf_steps,
                div_factor=1e2,
                final_div_factor=0.1,
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda_schedule)
        elif args.scheduler == "wsd":
            lambda_schedule = wsd_schedule(
                n_iterations=args.iterations,
                n_warmup=args.warmup_steps,
                fract_decay=args.wsd_fract_decay,
                init_div_factor=1e2,
                final_lr_factor=args.wsd_final_lr_scale,  # should be 0 here
                decay_type=args.decay_type,
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda_schedule)
        else:
            raise NotImplementedError(f"Unknown scheduler type: {args.scheduler}.")
    else:
        scheduler = None

    if (exp_dir / "ckpts" / "latest" / "main.pt").exists():
        if args.auto_resume:# Auto resume overwrites resume_from
            args.resume_from = str(exp_dir / "ckpts" / "latest")
        else:
            (exp_dir / "ckpts" / "latest" / "main.pt").unlink(missing_ok=True)
            # raise ValueError(
            #     f"The experiment dir {exp_dir} already exists. "
            #     + "To resume training, set auto_resume=True. "
            #     + "Otherwise, specify a different experiment name. "
            # )
    elif distributed_backend.is_master_process():
        exp_dir.mkdir(parents=True, exist_ok=True)

    if args.resume_from is None:
        if distributed_backend.is_master_process() and args.wandb:
            wandb.init(
                entity=args.wandb_entity,
                project=args.wandb_project,
                group=args.wandb_group,
                job_type=args.wandb_job_type,
                name=exp_name,
                config=vars(args),
            )
            wandb.define_metric("iter")
            wandb.define_metric("train/*", step_metric="iter")
            wandb.define_metric("val/*", step_metric="iter")
            wandb.define_metric("lr", step_metric="iter")
    else:
        if distributed_backend.is_master_process() and args.wandb:
            wandb_state = load_wandb_state(Path(args.resume_from))
            print(f'Initializing wandb on existing run id {wandb_state["id"]}')
            wandb.init(
                entity=wandb_state['entity'],
                project=wandb_state['project'],
                group=wandb_state['group'],
                job_type=wandb_state['job_type'],
                name=wandb_state['name'],
                config=vars(args),

                id=wandb_state['id'],
                resume="allow",)

    if distributed_backend.is_master_process() and args.wandb:
        wandb.log(
            {
                "parameters": params_cnt,
                "optimized_parameters": optimized_params_cnt,
                "non_embedding_parameters": nonemb_param_cnt,
            }
        )

    stats = train(
        model=model,
        opt=opt,
        datareaders=datareaders,
        scheduler=scheduler,
        exp_dir=exp_dir,
        distributed_backend=distributed_backend,
        cfg=args,
    )

    stats["args"] = vars(args)
    if distributed_backend.is_master_process():
        with open(exp_dir / "summary.json", "w") as fs:
            json.dump(stats, fs)
    distributed_backend.finalize()


def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--config_format", default="config", choices=config.registered_formats()
    )

    args, rem_args = parser.parse_known_args()

    return config.parse_args_with_format(
        format=args.config_format, base_parser=parser, args=rem_args, namespace=args
    )


def get_exp_name(args, distributed_backend):
    """Returns the name of the experiment, used for saving models and wandb."""
    if args.experiment_name is not None:
        return args.experiment_name

    rank = distributed_backend.rank

    exp_name = (
        f"{args.dataset}_{args.model}_nlayers{args.n_layer}"
        f"_nhead{args.n_head}_lr{args.lr}"
        f"_sched_{args.scheduler}_warmup{args.warmup_steps}"
        f"_decay_{args.decay_type}_{args.wsd_fract_decay}"
        f"_iter{args.iterations}"
        f"_bs{args.batch_size}x{args.acc_steps}_ws{args.world_size}"
    )
    # for mup
    if args.model == "mup_noam":
        exp_name = (
            f"{args.dataset}_{args.model}"
            f"_opt{args.opt}"
            f"_nlayers{args.n_layer}"
            # f"_nhead{args.n_head}"
            f"_lr{args.lr}"
            f"_sched_{args.scheduler}"
            f"_decay_{args.decay_type}"
            # f"_warmup{args.warmup_steps}"
            f"_iter{args.iterations}"
            f"_init{args.init_std}_sce{args.scale_emb}"
            f"_scd{args.scale_depth}"
            # f"_bs{args.batch_size}x{args.acc_steps}_ws{args.world_size}"
        )
    if args.wandb_run_prefix != "none":
        exp_name = args.wandb_run_prefix + "_" + exp_name
    exp_name += f"_seed{args.seed - rank}"
    exp_name += f"_data_seed{args.data_seed}"

    if args.weight_average:
        exp_name += f"_WA"
    if args.opt == "SFAdamW":
        exp_name += f"_beta1_{args.beta1}"
        exp_name += f"_beta2_{args.beta2}"
    return exp_name


def get_data_readers(args, verbose=True):
    data_srcs = get_dataset(args)
    train_reader = DataReader(
        data_src=data_srcs["train"],
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        seed=args.data_seed,
        with_replacement=False,
        auto_shard=True,
        keep_in_ram=args.data_in_ram,
    )
    val_reader = DataReader(
        data_src=data_srcs["val"],
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        seed=args.data_seed,
        with_replacement=False,
        auto_shard=False,  # NOTE Identical Per Rank
        keep_in_ram=args.data_in_ram,
    )

    if verbose:
        print(f"Num training tokens: {train_reader.num_tokens}")
        print(f"Num validation tokens: {val_reader.num_tokens}")

    return {
        "train": train_reader,
        "val": val_reader,
    }


if __name__ == "__main__":
    args = get_args()
    main(args)
