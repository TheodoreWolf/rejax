"""Tune MPO on Brax environments with Weights & Biases sweeps."""

import argparse
from collections.abc import Mapping

import jax
import numpy as np

import wandb
from rejax import MPO


DEFAULT_CONFIG = {
    "env": "brax/hopper",
    "backend": "mjx",
    "activation": "tanh",
    "hidden_layer_sizes": "256,256",
    "total_timesteps": 1_000_000,
    "eval_freq": 50_000,
    "num_envs": 128,
    "buffer_size": 250_000,
    "fill_buffer": 8_192,
    "batch_size": 512,
    "num_epochs": 4,
    "learning_rate": 3e-4,
    "dual_learning_rate": 1e-2,
    "gamma": 0.99,
    "max_grad_norm": 10.0,
    "num_action_samples": 20,
    "policy_eval_num_val_samples": 128,
    "target_network_update_freq": 100,
    "polyak": 0.0,
    "epsilon": 0.1,
    "epsilon_mean": 0.0025,
    "epsilon_stddev": 1e-6,
    "epsilon_penalty": 0.001,
    "action_penalization": True,
    "per_dim_constraining": True,
    "normalize_observations": True,
    "skip_initial_evaluation": True,
    "seed": 0,
    "num_seeds": 3,
}


def parse_hidden_layer_sizes(value):
    if isinstance(value, str):
        return tuple(int(size.strip()) for size in value.split(",") if size.strip())
    return tuple(int(size) for size in value)


def build_algo_config(config: Mapping):
    config = dict(config)
    seed = int(config.pop("seed"))
    num_seeds = int(config.pop("num_seeds"))
    backend = config.pop("backend")
    activation = config.pop("activation")
    hidden_layer_sizes = parse_hidden_layer_sizes(config.pop("hidden_layer_sizes"))

    config["env_params"] = {"backend": backend}
    config["agent_kwargs"] = {"activation": activation}
    config["hidden_layer_sizes"] = hidden_layer_sizes

    int_keys = {
        "batch_size",
        "buffer_size",
        "eval_freq",
        "fill_buffer",
        "num_action_samples",
        "num_envs",
        "num_epochs",
        "policy_eval_num_val_samples",
        "target_network_update_freq",
        "total_timesteps",
    }
    for key in int_keys:
        config[key] = int(config[key])

    return seed, num_seeds, config


def log_evaluations(lengths, returns, config):
    lengths = np.asarray(lengths)
    returns = np.asarray(returns)

    if config["skip_initial_evaluation"]:
        steps = np.arange(1, returns.shape[1] + 1) * config["eval_freq"]
    else:
        steps = np.arange(returns.shape[1]) * config["eval_freq"]

    eval_lengths_by_step = lengths.swapaxes(0, 1)
    eval_returns_by_step = returns.swapaxes(0, 1)
    for step, eval_lengths, eval_returns in zip(
        steps, eval_lengths_by_step, eval_returns_by_step
    ):
        length_by_seed = eval_lengths.mean(axis=-1)
        return_by_seed = eval_returns.mean(axis=-1)
        data = {
            "eval/episode_length_mean": float(length_by_seed.mean()),
            "eval/episode_length_std": float(length_by_seed.std()),
            "eval/return_mean": float(return_by_seed.mean()),
            "eval/return_std": float(return_by_seed.std()),
        }
        wandb.log(data, step=int(step))

    wandb.run.summary["final_eval/episode_length_mean"] = data[
        "eval/episode_length_mean"
    ]
    wandb.run.summary["final_eval/episode_length_std"] = data["eval/episode_length_std"]
    wandb.run.summary["final_eval/return_mean"] = data["eval/return_mean"]
    wandb.run.summary["final_eval/return_std"] = data["eval/return_std"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="rejax-benchmark")
    parser.add_argument("--entity", default="flair")
    args, _ = parser.parse_known_args()

    wandb.init(project=args.project, entity=args.entity, config=DEFAULT_CONFIG)
    seed, num_seeds, config = build_algo_config(wandb.config)

    algo = MPO.create(**config)
    rngs = jax.random.split(jax.random.PRNGKey(seed), num_seeds)

    print(f"Compiling MPO training over {num_seeds} seeds...")
    train = jax.jit(jax.vmap(algo.train)).lower(rngs).compile()
    backend = config["env_params"]["backend"]
    print(f"Training MPO on {config['env']} with backend={backend}...")
    _, (lengths, returns) = train(rngs)
    log_evaluations(lengths, returns, config)
    wandb.finish()


if __name__ == "__main__":
    main()
