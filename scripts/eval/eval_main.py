#本文件是运行的主函数，可更改内容如下：
'''
1.    # ===== 1. Trajectory Client =====
    traj_client = Gr00tTrajectoryClient(
        url=f"http://{args.gr00t_host}:{args.gr00t_port}/act"
    )
    这部分是连接http server的方式，创建一个自己的client（例如已有的gr00t的InternNav/internnav/evaluator/HTTPTrajectoryClient.py）。
    从其中声明函数类在这里实现即可
2.    # * 2. initialize evaluator
    # ===== 1. 构建 Agent=====
    agent = LLMAgent(
        traj_client=traj_client,
        processor=None,   # Gr00t 不需要 processor
        args=args,
        device="cuda",
    )

    # ===== 2. 构建 Evaluator =====
    evaluator = Evaluator(
        config_path=args.habitat_config_path,
        split=args.eval_split,
        output_path=args.output_path,
        args=args,
        agent=agent,
        max_steps=500,
        idx=get_rank(),
        env_num=get_world_size(),
    )
    这部分是构建agent及Evaluator，便于之后的运行，相关参数大体更换为新的即可。
    如果有新的要声明的参数在InternNav/internnav/evaluator/final_habitat_vln_evaluator.py声明
'''
import argparse
import os
import numpy as np

from internnav.evaluator.final_habitat_vln_evaluator import Evaluator
from internnav.evaluator.final_habitat_vln_evaluator import LLMAgent
from internnav.evaluator.HTTPTrajectoryClient import Gr00tTrajectoryClient

def parse_args():

    parser = argparse.ArgumentParser(description='Evaluate InternVLA-N1 on Habitat')
    parser.add_argument("--mode", default='dual_system', type=str, help="inference mode: dual_system or system2")
    parser.add_argument("--model_path", type=str, default="")
    # 如果后面habitat这边的数据集和路径什么的要变化，改下面这个文件
    parser.add_argument("--habitat_config_path", type=str, default='scripts/eval/configs/vln_r2r_no_oracle.yaml')
    parser.add_argument("--eval_split", type=str, default='val_unseen')
    parser.add_argument("--output_path", type=str, default='./logs/habitat/test')  #!
    parser.add_argument("--num_future_steps", type=int, default=4)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument(
        "--save-snapshots",
        action="store_true",
        help="Save detailed, non-essential per-step metrics and episode videos.",
    )
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument(
        "--init_look_down_steps",
        type=int,
        default=int(os.getenv("HABITAT_INIT_LOOK_DOWN_STEPS", "2")),
        help=(
            "Number of LOOK_DOWN actions after episode reset before evaluation starts. "
            "Use 0 to keep the default Habitat camera pitch."
        ),
    )
    parser.add_argument("--resize_w", type=int, default=384)
    parser.add_argument("--resize_h", type=int, default=384)
    parser.add_argument("--predict_step_nums", type=int, default=16)
    parser.add_argument(
        "--success_distance",
        type=float,
        default=float(os.getenv("HABITAT_SUCCESS_DISTANCE", "1.0")),
        help="Distance threshold in meters used by Habitat success/SPL metrics.",
    )
    parser.add_argument("--continuous_traj", action="store_true", default=False)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--max_eval_episodes", type=int, default=0, help="0 means evaluate all episodes")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=int(os.getenv("HABITAT_EVAL_MAX_STEPS", "500")),
        help="Maximum Habitat actions per episode.",
    )
    parser.add_argument(
        "--eval_episode_ids",
        type=str,
        default="",
        help="Comma-separated Habitat episode ids to evaluate. Empty means all episodes.",
    )
    parser.add_argument(
        "--manual_instruction",
        type=str,
        default="",
        help=(
            "Override the dataset instruction for every selected episode. "
            "Use this for manual rollouts only; goal metrics will still follow the original dataset episode."
        ),
    )
    parser.add_argument(
        "--random_eval_episodes",
        action="store_true",
        default=False,
        help="Shuffle candidate episodes before applying --max_eval_episodes.",
    )
    parser.add_argument(
        "--eval_seed",
        type=int,
        default=0,
        help="Random seed used by --random_eval_episodes.",
    )
    parser.add_argument("--shard_rank", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument(
        "--exclude_episode_ids_file",
        type=str,
        default="",
        help="JSON file of completed scene_id/episode_id pairs to exclude before sharding.",
    )

    parser.add_argument('--sim_gpu', default=int(os.getenv("HABITAT_SIM_GPU", "5")), type=int, help='Habitat-Sim renderer GPU id')

    ###连接server_Gr00t，后续使用只需要换成正确的url即可，见main()的 ===== 1. Trajectory Client =====
    parser.add_argument('--gr00t_host', default='127.0.0.1')
    parser.add_argument("--navigation-task", choices=("vln", "tracking"), default="vln")
    parser.add_argument('--gr00t_port', default=9000, type=int)  

    return parser.parse_args()

def main():
    args = parse_args()
    print("[[[[args.mode]]]]", args.mode)

    if args.num_shards <= 0 or not 0 <= args.shard_rank < args.num_shards:
        raise ValueError("Expected 0 <= shard_rank < num_shards")
    np.random.seed(args.shard_rank)

    # ===== 1. Trajectory Client =====
    print(f"Connecting to Trajectory Server at http://{args.gr00t_host}:{args.gr00t_port}/act ...")
    
    # 使用新写的 Client
    # traj_client = InternVATrajectoryClient(
    #     url=f"http://{args.gr00t_host}:{args.gr00t_port}/act",
    #     max_history=args.num_history # 传入历史长度参数
    # )
    traj_client = Gr00tTrajectoryClient(
        url=f"http://{args.gr00t_host}:{args.gr00t_port}/act",
        env_id=f"habitat-shard-{args.shard_rank}",
        navigation_task=args.navigation_task,
        debug_output_path=os.path.join(args.output_path, "enactive_server_snapshots"),
    )

    # * 2. initialize evaluator
    # ===== 1. 构建 Agent=====
    agent = LLMAgent(
        traj_client=traj_client,
        processor=None,   # Gr00t 不需要 processor
        args=args,
        device="cuda",
    )

    # ===== 2. 构建 Evaluator =====
    evaluator = Evaluator(
        config_path=args.habitat_config_path,
        split=args.eval_split,
        output_path=args.output_path,
        args=args,
        agent=agent,
        max_steps=args.max_steps,
        idx=args.shard_rank,
        env_num=args.num_shards,
    )

    # ===== 3. 运行评测 =====  这部分一般不用改，除非有什么最后要加的除推理之外的逻辑，计算部分已经放在了Evaluator()
    evaluator.run()

if __name__ == "__main__":
    main()
