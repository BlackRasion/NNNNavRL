import argparse
import json
import math
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class PsoConfig:
    num_particles: int
    num_iters: int
    inertia: float
    c1: float
    c2: float
    seed: int


# 每个待优化权重定义为：名称、下界、上界、默认值
WEIGHT_SPECS: List[Tuple[str, float, float, float]] = [
    ("reward_distance_2d", 0.0, 2.0, 0.35),
    ("reward_progress_2d", 0.0, 5.0, 2.5),
    ("reward_velocity_2d", 0.0, 4.0, 2.0),
    ("reward_heading_2d", 0.0, 2.0, 0.5),
    ("safety_penalty_static_2d", 0.0, 8.0, 3.0),
    ("safety_penalty_dynamic_2d", 0.0, 8.0, 3.0),
    ("angular_penalty_2d", 0.0, 3.0, 1.0),
    ("collision_penalty_2d", 0.1, 4.0, 1.0),
    ("goal_reward_2d", 0.1, 4.0, 1.0),
    ("time_penalty_2d", 0.0, 4.0, 1.0),
]


def build_hydra_weight_overrides(weights: Dict[str, float]) -> List[str]:
    overrides = []
    for name, value in weights.items():
        overrides.append(f"reward_weights.{name}={value:.6f}")
    return overrides


def find_newest_wandb_summary(wandb_dir: Path, after_ts: float) -> Path:
    run_dirs = [p for p in wandb_dir.glob("offline-run-*") if p.is_dir()]
    run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for run_dir in run_dirs:
        if run_dir.stat().st_mtime < after_ts:
            continue
        summary = run_dir / "files" / "wandb-summary.json"
        if summary.exists():
            return summary
    raise FileNotFoundError("未找到本轮训练生成的 wandb-summary.json")


def read_metrics_from_summary(summary_file: Path) -> Dict[str, float]:
    with summary_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    keys = [
        "train/stats.collision",
        "train/stats.reach_goal",
        "train/stats.episode_len",
        "train/stats.return",
    ]
    missing = [k for k in keys if k not in data]
    if missing:
        raise KeyError(f"summary 缺少关键指标: {missing}")
    return {
        "collision": float(data["train/stats.collision"]),
        "reach_goal": float(data["train/stats.reach_goal"]),
        "episode_len": float(data["train/stats.episode_len"]),
        "return": float(data["train/stats.return"]),
    }


def fitness(metrics: Dict[str, float]) -> float:
    # 目标：高到达率、低碰撞率、较短回合长度；return 仅作为弱辅助项
    return (
        5.0 * metrics["reach_goal"]
        - 6.0 * metrics["collision"]
        - 0.002 * metrics["episode_len"]
        + 0.0001 * metrics["return"]
    )


def _read_text_tail(file_path: Path, max_bytes: int = 16384, max_chars: int = 4000) -> str:
    if not file_path.exists():
        return "<log file missing>"
    with file_path.open("rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - max_bytes), 0)
        data = f.read()
    text = data.decode("utf-8", errors="replace")
    return text[-max_chars:]


def _safe_unlink(file_path: Path) -> None:
    try:
        file_path.unlink(missing_ok=True)
    except Exception:
        pass


def run_short_training(
    scripts_dir: Path,
    weights: Dict[str, float],
    max_frame_num: int,
    num_envs: int,
    timeout_sec: int,
) -> Dict[str, float]:
    cmd = [
        "python",
        "train.py",
        "headless=True",
        "wandb.mode=offline",
        "wandb.run_id=null",
        f"max_frame_num={max_frame_num}",
        f"env.num_envs={num_envs}",
        "eval_interval=1000000000",
        "save_interval=1000000000",
    ]
    cmd.extend(build_hydra_weight_overrides(weights))
    wandb_dir = scripts_dir / "wandb"
    log_dir = scripts_dir / "pso_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    run_tag = f"{int(time.time() * 1000)}_{random.randint(0, 999999):06d}"
    stdout_log = log_dir / f"pso_train_{run_tag}.stdout.log"
    stderr_log = log_dir / f"pso_train_{run_tag}.stderr.log"

    before = time.time()
    try:
        with stdout_log.open("w", encoding="utf-8", errors="replace") as out_f, stderr_log.open(
            "w", encoding="utf-8", errors="replace"
        ) as err_f:
            proc = subprocess.run(
                cmd,
                cwd=str(scripts_dir),
                stdout=out_f,
                stderr=err_f,
                text=True,
                timeout=timeout_sec,
                check=False,
            )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            "训练子进程超时。\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout:\n{_read_text_tail(stdout_log)}\n"
            f"stderr:\n{_read_text_tail(stderr_log)}"
        )

    if proc.returncode != 0:
        raise RuntimeError(
            "训练子进程失败。\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout:\n{_read_text_tail(stdout_log)}\n"
            f"stderr:\n{_read_text_tail(stderr_log)}"
        )
    summary = find_newest_wandb_summary(wandb_dir, before)
    metrics = read_metrics_from_summary(summary)
    _safe_unlink(stdout_log)
    _safe_unlink(stderr_log)
    return metrics


def sample_initial_particle(rng: random.Random) -> Dict[str, float]:
    particle = {}
    for name, low, high, _ in WEIGHT_SPECS:
        particle[name] = rng.uniform(low, high)
    return particle


def default_particle() -> Dict[str, float]:
    return {name: default for name, _, _, default in WEIGHT_SPECS}


def clip_particle(particle: Dict[str, float]) -> None:
    bound = {name: (low, high) for name, low, high, _ in WEIGHT_SPECS}
    for name, value in list(particle.items()):
        low, high = bound[name]
        particle[name] = min(high, max(low, value))


def as_weight_line(weights: Dict[str, float]) -> str:
    ordered = [f"{name}={weights[name]:.4f}" for name, _, _, _ in WEIGHT_SPECS]
    return ", ".join(ordered)


def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj):
            return None
        if math.isinf(obj):
            return None
        return obj
    return obj


def optimize(
    scripts_dir: Path,
    pso: PsoConfig,
    max_frame_num: int,
    num_envs: int,
    timeout_sec: int,
    out_json: Path,
) -> None:
    rng = random.Random(pso.seed)

    particles: List[Dict[str, float]] = [sample_initial_particle(rng) for _ in range(pso.num_particles)]
    particles[0] = default_particle()
    velocities: List[Dict[str, float]] = [{name: 0.0 for name, _, _, _ in WEIGHT_SPECS} for _ in particles]
    pbest = [dict(p) for p in particles]
    pbest_score = [-math.inf for _ in particles]
    gbest = dict(particles[0])
    gbest_score = -math.inf
    history = []

    for it in range(pso.num_iters):
        print(f"\n[PSO] Iteration {it + 1}/{pso.num_iters}")
        for i, particle in enumerate(particles):
            try:
                metrics = run_short_training(
                    scripts_dir=scripts_dir,
                    weights=particle,
                    max_frame_num=max_frame_num,
                    num_envs=num_envs,
                    timeout_sec=timeout_sec,
                )
                score = fitness(metrics)
            except Exception as e:
                metrics = {"collision": 1.0, "reach_goal": 0.0, "episode_len": float("inf"), "return": -1e9}
                score = -1e12
                print(f"[PSO] particle={i} 训练失败，记极低分。error={e}")

            if score > pbest_score[i]:
                pbest_score[i] = score
                pbest[i] = dict(particle)
            if score > gbest_score:
                gbest_score = score
                gbest = dict(particle)

            row = {
                "iter": it,
                "particle": i,
                "score": score,
                "metrics": metrics,
                "weights": dict(particle),
            }
            history.append(row)
            print(
                f"[PSO] particle={i} score={score:.4f} "
                f"reach_goal={metrics['reach_goal']:.4f} collision={metrics['collision']:.4f} "
                f"episode_len={metrics['episode_len']:.2f}"
            )

        for i, particle in enumerate(particles):
            r1 = rng.random()
            r2 = rng.random()
            for name, low, high, _ in WEIGHT_SPECS:
                v = velocities[i][name]
                x = particle[name]
                v = (
                    pso.inertia * v
                    + pso.c1 * r1 * (pbest[i][name] - x)
                    + pso.c2 * r2 * (gbest[name] - x)
                )
                velocities[i][name] = v
                particle[name] = x + v
            clip_particle(particle)

        print(f"[PSO] iter={it + 1} 当前全局最优 score={gbest_score:.4f}")
        print(f"[PSO] gbest: {as_weight_line(gbest)}")

    out = {
        "best_score": gbest_score,
        "best_weights": gbest,
        "history": history,
        "pso": pso.__dict__,
        "max_frame_num": max_frame_num,
        "num_envs": num_envs,
    }
    out_sanitized = sanitize_for_json(out)
    out_json.write_text(
        json.dumps(out_sanitized, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"\n[PSO] 完成。最优 score={gbest_score:.6f}")
    print(f"[PSO] 最优权重: {as_weight_line(gbest)}")
    print(f"[PSO] 结果已保存: {out_json}")


def main():
    parser = argparse.ArgumentParser(description="PSO 自动优化 NavRL 奖励权重")
    parser.add_argument("--repo-root", type=str, default=r"d:\Code\NNNNavRL")
    parser.add_argument("--num-particles", type=int, default=8)
    parser.add_argument("--num-iters", type=int, default=6)
    parser.add_argument("--inertia", type=float, default=0.72)
    parser.add_argument("--c1", type=float, default=1.49)
    parser.add_argument("--c2", type=float, default=1.49)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-frame-num", type=int, default=200000)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--timeout-sec", type=int, default=7200)
    parser.add_argument(
        "--out-json",
        type=str,
        default=r"d:\Code\NNNNavRL\isaac-training\training\scripts\pso_reward_result.json",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    scripts_dir = repo_root / "isaac-training" / "training" / "scripts"
    out_json = Path(args.out_json)

    pso = PsoConfig(
        num_particles=args.num_particles,
        num_iters=args.num_iters,
        inertia=args.inertia,
        c1=args.c1,
        c2=args.c2,
        seed=args.seed,
    )

    optimize(
        scripts_dir=scripts_dir,
        pso=pso,
        max_frame_num=args.max_frame_num,
        num_envs=args.num_envs,
        timeout_sec=args.timeout_sec,
        out_json=out_json,
    )


if __name__ == "__main__":
    main()
