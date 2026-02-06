from contextlib import nullcontext
import copy
from pathlib import Path
import time
import yaml
import sys
import os
import torch
import torch.distributed as dist
import wandb
import gpustat
import socket
from tqdm import tqdm

from logger.logger import DynamicsLogger
from optim.weight_averaging import (
    WeightAverager,
    eval_ema,
    eval_wa,
    ExponentialWeightAverager,
)
from .utils import (
    eval,
    get_batch,
    load_checkpoint,
    load_worker_state,
    save_checkpoint,
    save_worker_state,
    save_wandb_state,
    ElapsedTime,
)

def get_gpu_mem_usage(ddp=True):
    gpus = gpustat.new_query().gpus
    gids = list(range(dist.get_world_size())) if ddp else list(map(int, os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
    # gids = list(map(int, os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
    mem_per_gpu = [
        int(proc['gpu_memory_usage']) / 1000. # MB to GB
        for gid in gids
        for proc in gpus[gid]['processes']
    ]
    avg = round(sum(mem_per_gpu) / len(mem_per_gpu), 2)
    M = round(max(mem_per_gpu), 2)
    m = round(min(mem_per_gpu), 2)
    return avg, M, m

def train(
    model,
    opt,
    datareaders,
    scheduler,
    exp_dir,
    distributed_backend,
    cfg,
):
    not_compiled_model = model
    if cfg.compile:
        print(f"Compiling model ...")
        model = torch.compile(model)

    if "cuda" in cfg.device:
        type_ctx = torch.amp.autocast(
            device_type="cuda",
            dtype={
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }[cfg.dtype],
        )
    else:
        type_ctx = nullcontext()

    if cfg.resume_from:
        # This is a full resume including the model weights, optimizer, state
        # dataloader state, random seed, etc. Not indended for fine tuning or
        # other scenarios where some of these should change.
        ckpt_dir = Path(cfg.resume_from)
        curr_iter = load_checkpoint(
            model,
            opt,
            scheduler,
            ckpt_dir,
            cfg.device,
        )
        print(f"\nResuming training from iteration {curr_iter} @ {cfg.resume_from}")
        # model is already trained at step curr_iter => advance by 1 to avoid overwriting the latest checkpoint in the first 2 if statements from while-loop
        curr_iter += 1 # advance by 1 because the checkpoint is already trained at
        load_worker_state(ckpt_dir)
    else:
        curr_iter = 0

    if cfg.weight_average:
        # This does generally not support resuming training, but will work if
        # cfg.wa_interval perfectly divides the iteration number of the chkpt.
        # Otherwise, the first avg will not be correctly computed, with a bias
        # towards the first sample and missing values for earlier iterations.
        weight_averager = WeightAverager(
            not_compiled_model,
            horizon=cfg.wa_horizon,
            interval=cfg.wa_interval,
            save_dir=None if cfg.wa_use_temp_dir else exp_dir / "avgs",
            dtype={
                "float32": torch.float32,
                "float64": torch.float64,
            }[cfg.wa_dtype],
            count=curr_iter,
        )

    if cfg.exponential_moving_average:
        ema = ExponentialWeightAverager(
            not_compiled_model,
            interval=cfg.ema_interval,
            decay=cfg.ema_decay,
            warmup=cfg.warmup_steps if cfg.ema_after_warmup else 0,
            dtype={
                "float32": torch.float32,
                "float64": torch.float64,
            }[cfg.wa_dtype],
        )

    if distributed_backend.is_master_process() and cfg.log_dynamics:
        with open(cfg.dynamics_logger_cfg, "r") as f:
            dlcfg = yaml.safe_load(f)

        # Hooks into optimizer
        dlogger = DynamicsLogger(
            model, opt, dlcfg, cfg.results_base_folder, wandb=cfg.wandb
        )
        dlogger.iteration = curr_iter

    substep = curr_iter * cfg.acc_steps
    train_reader, val_reader = datareaders["train"], datareaders["val"]
    train_reader.set_step(substep)
    stats = {"train_loss": [], "val_loss": [], "val_pp": [], "val_acc": []}
    model.train()

    # Initialize the progress bar
    if distributed_backend.is_master_process():
        pbar = tqdm(total=cfg.iterations, desc="Training Progress", position=curr_iter)
    else:
        pbar = None

    gradient_accumulator = {}
    save_gradients = False

    # print(f'\t[optim.py]rank = {dist.get_rank()}: batch-size = {cfg.batch_size}, acc-steps = {cfg.acc_steps}')

    elapsed: ElapsedTime = ElapsedTime()
    if distributed_backend.is_master_process():
        ar_elapsed: ElapsedTime = ElapsedTime() # "ar" means "all-reduced

    if cfg.opt in ['dash-lw']:
        opt.log_bucket_stats(path=cfg.results_base_folder)


    while curr_iter <= cfg.iterations:
        if curr_iter == cfg.limit_iters_to:
            print(f'Reached iteration limit of {curr_iter}')
            break

        # Save permanent checkpoint
        if cfg.permanent_ckpt_interval > 0:
            if curr_iter % cfg.permanent_ckpt_interval == 0:
                ckpt_dir = exp_dir / "ckpts" / str(curr_iter)

                is_master = distributed_backend.is_master_process()
                save_checkpoint(model, opt, scheduler, curr_iter, ckpt_dir, is_master)
                save_worker_state(ckpt_dir)
                save_wandb_state(ckpt_dir, is_master)
                print(f'Saved "permanent" checkpoint @ step {curr_iter}')

        # Save temporary checkpoint for resuming training
        if cfg.latest_ckpt_interval > 0:
            if curr_iter % cfg.latest_ckpt_interval == 0 or curr_iter == cfg.iterations:
                ckpt_dir = exp_dir / "ckpts" / "latest"

                is_master = distributed_backend.is_master_process()
                save_checkpoint(model, opt, scheduler, curr_iter, ckpt_dir, is_master)
                save_worker_state(ckpt_dir)
                save_wandb_state(ckpt_dir, is_master)
                print(f'Saved "latest" checkpoint @ step {curr_iter}')

        ws = distributed_backend.get_world_size()
        tokens = ws * substep * cfg.sequence_length * cfg.batch_size
        epoch = tokens / train_reader.num_tokens
        if (
            curr_iter % cfg.eval_interval == 0
            or curr_iter == cfg.iterations
            or (curr_iter in cfg.full_eval_at)
        ):

            eval_and_log(
                curr_iter,
                epoch,
                model,
                val_reader,
                type_ctx,
                distributed_backend,
                cfg,
                opt,
                full_eval=(curr_iter in cfg.full_eval_at),
            )

            if curr_iter > cfg.wa_interval and cfg.weight_average:
                eval_wa(
                    curr_iter,
                    not_compiled_model,
                    weight_averager,
                    val_reader,
                    type_ctx,
                    distributed_backend,
                    cfg,
                    full_eval=(curr_iter in cfg.full_eval_at),
                )
            if cfg.exponential_moving_average:
                eval_ema(
                    curr_iter,
                    not_compiled_model,
                    ema,
                    val_reader,
                    type_ctx,
                    distributed_backend,
                    cfg,
                    full_eval=(curr_iter in cfg.full_eval_at),
                )

        if curr_iter == cfg.iterations:
            # Save checkpoints and evaluate at final iteration, but no need to train further
            break

        # Train model
        t_start = time.perf_counter_ns()
        elapsed_fw_step = 0
        elapsed_bw_step = 0
        elapsed_opt_step = 0
        for microstep_idx in range(cfg.acc_steps):  # gradient accumulation
            x, y = get_batch(train_reader, device=cfg.device)
            with type_ctx:
                with distributed_backend.get_context_for_microstep_forward(
                    model=model,
                    microstep_idx=microstep_idx,
                    gradient_accumulation_steps=cfg.acc_steps,
                ):
                    torch.cuda.synchronize(cfg.device)
                    t_fw_start = time.perf_counter_ns()
                    outputs = model(x, targets=y)
                    torch.cuda.synchronize(cfg.device)
                    t_fw_end = time.perf_counter_ns()
                    elapsed_fw_step += (t_fw_end - t_fw_start) / 1e9

            loss = outputs["loss"] / cfg.acc_steps

            torch.cuda.synchronize(cfg.device)
            t_bw_start = time.perf_counter_ns()
            loss.backward()
            torch.cuda.synchronize(cfg.device)
            t_bw_end = time.perf_counter_ns()
            elapsed_bw_step += (t_bw_end - t_bw_start) / 1e9
            substep += 1
        # end for grad accum

        if cfg.grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        torch.cuda.synchronize(cfg.device)
        t_opt_step_start = time.perf_counter_ns()
        opt.step()

        if cfg.opt in ['dash-lw'] and cfg.shmp_log_stats_interval > 0 and (curr_iter == 1 or curr_iter % cfg.shmp_log_stats_interval == 0):
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    opt.log_layer_stats()
            else:
                opt.log_layer_stats()

        torch.cuda.synchronize(cfg.device)
        t_opt_step_end = time.perf_counter_ns()
        elapsed_opt_step = (t_opt_step_end - t_opt_step_start) / 1e9

        # all-reduce for FW time
        tensor_fw_step = torch.tensor(elapsed_fw_step, dtype=torch.double, device=cfg.device)
        if dist.is_initialized():
            dist.all_reduce(tensor_fw_step, op=dist.ReduceOp.MAX)

        # all-reduce for BW time
        tensor_bw_step = torch.tensor(elapsed_bw_step, dtype=torch.double, device=cfg.device)
        if dist.is_initialized():
            dist.all_reduce(tensor_bw_step, op=dist.ReduceOp.MAX)

        # all-reduce for opt-step time
        tensor_opt_step = torch.tensor(elapsed_opt_step, dtype=torch.double, device=cfg.device)
        if dist.is_initialized():
            dist.all_reduce(tensor_opt_step, op=dist.ReduceOp.MAX)

        # update ElapsedTime objects:
        # the local one on each worker
        elapsed.update(elapsed_fw=elapsed_fw_step, elapsed_bw=elapsed_bw_step, elapsed_opt=elapsed_opt_step)

        # the one aggregated one
        if distributed_backend.is_master_process():
            ar_elapsed.update(elapsed_fw=tensor_fw_step.item(), elapsed_bw=tensor_bw_step.item(), elapsed_opt=tensor_opt_step.item())
            ar_elapsed.log_wandb(curr_iter, stdout=False)

        scheduler.step()
        opt.zero_grad(set_to_none=True)

        if cfg.weight_average:
            weight_averager.step(
                not_compiled_model, distributed_backend.is_master_process()
            )
        if cfg.exponential_moving_average:
            ema.step(not_compiled_model, distributed_backend.is_master_process())
        dt = (time.perf_counter_ns() - t_start) / 1e9

        curr_iter += 1
        if distributed_backend.is_master_process():
            pbar.update(1)

        if (
            cfg.log_interval
            and curr_iter % cfg.log_interval == 0
            and distributed_backend.is_master_process()  # Only log on master rank
        ):
            train_loss = loss.detach().cpu().item() * cfg.acc_steps

            current_lrs = [param_group["lr"] for param_group in opt.param_groups]

            # model_sparsities = compute_layer_sparsities(model)

            print(
                f"Train: Iter={curr_iter} ({epoch:0.3f} epochs) "
                f"train_loss={train_loss:.3f} iter_dt={dt:.2e}s "
                f"lr={current_lrs[0]:.2e} "
                # f"sparsity={model_sparsities}"
            )

            if torch.isnan(loss):
                print(f'Found NaN in loss!!!')
                break

            if cfg.wandb:
                avg, M, m = get_gpu_mem_usage(ddp=dist.is_initialized())
                wandb_data = {
                    "iter": curr_iter,
                    "train/loss": train_loss,
                    "train/perplexity": 2.71828**train_loss,
                    "lr": current_lrs[0],
                    # "sparsity": model_sparsities,
                    "iter_dt": dt,
                    "elapsed_total": dt,
                    "elapsed_opt_step": elapsed_opt_step,
                    "elapsed_fw": elapsed_fw_step,
                    "elapsed_bw": elapsed_bw_step,

                    'gpu_mem_avg': avg,
                    'gpu_mem_max': M,
                    'gpu_mem_min': m,
                }

                wandb.log(wandb_data)
    # end while
    print(f'Training finished')
    return stats

def compute_layer_sparsities(model):
    sparsities = set()
    for p in model.parameters():
        sp = str(int((p == 0).sum().item() / p.numel() * 100.))
        sparsities.add(sp)
    return ';'.join(sparsities)

def eval_and_log(
    curr_iter,
    epoch,
    model,
    val_reader,
    type_ctx,
    distributed_backend,
    cfg,
    opt,
    full_eval=False,
):
    if not distributed_backend.is_master_process():
        # Only evaluate and log on master rank
        return

    model.eval()
    if cfg.opt == "SFAdamW":
        opt.eval()

    if curr_iter == cfg.iterations or full_eval:
        max_num_batches = val_reader.num_batches()
    else:
        max_num_batches = cfg.eval_batches

    # to make sure we start from the beginning of the validation set,
    # i.e. repeat the same batches
    val_reader.set_step(0)
    val_acc, val_loss, val_perplexity = eval(
        model,
        val_reader,
        cfg.device,
        max_num_batches=max_num_batches,
        ctx=type_ctx,
        cfg=cfg,
    )

    print(
        f">Eval: Iter={curr_iter} ({epoch:0.3f} epochs) "
        f"val_loss={val_loss:.3f} "
        f"val_pp={val_perplexity:.3f} "
        f"val_acc={val_acc:3f}"
    )

    if cfg.wandb:
        if curr_iter == cfg.iterations or full_eval:
            logs = {
                "iter": curr_iter,
                "final-val/loss": val_loss,
                "final-val/perplexity": val_perplexity,
                "final-val/acc": val_acc,
            }
        else:
            logs = {
                "iter": curr_iter,
                "val/loss": val_loss,
                "val/perplexity": val_perplexity,
                "val/acc": val_acc,
            }

        wandb.log(logs)
        if cfg.eval_seq_prefix != "none" and (
            curr_iter % (cfg.eval_interval * 5) == 0 or curr_iter == cfg.iterations
        ):
            text_table = wandb.Table(columns=["itr", "val-pp", "text"])

            out_str = distributed_backend.get_raw_model(model).generate_from_string(
                cfg.eval_seq_prefix,
                max_new_tokens=40,
                temperature=0.9,
                top_k=None,
            )
            text_table.add_data(curr_iter, val_perplexity, out_str)
            # why a copy? see github.com/wandb/wandb/issues/2981
            wandb.log({f"generated-text-{wandb.run.name}": copy.copy(text_table)})
    model.train()
