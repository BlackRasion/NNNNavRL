import argparse
import itertools
import json
import pathlib


def build_grid():
    return {
        "reward_progress_scale": [8.0, 10.0, 12.0],
        "reward_velocity_scale": [1.5, 2.0, 2.5],
        "reward_safety_static_scale": [1.5, 2.0, 2.5],
        "reward_safety_dynamic_scale": [2.5, 3.0, 3.5],
        "reward_heading_scale": [0.25, 0.5, 0.75],
        "collision_penalty": [-150.0, -200.0, -250.0],
        "goal_reward": [150.0, 200.0, 250.0],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="reward_sweep.jsonl")
    args = parser.parse_args()

    grid = build_grid()
    keys = list(grid.keys())
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as f:
        for values in itertools.product(*(grid[k] for k in keys)):
            item = dict(zip(keys, values))
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(str(output.resolve()))


if __name__ == "__main__":
    main()
