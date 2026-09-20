#本项目以Gr00t轨迹预测模型为例，通过 HTTP Trajectory Server 的方式接入 InternNav 的 Habitat VLN 评测流程
#便于后期更换模型接入接口等
主函数：InternNav/scripts/eval/eval_main.py 用于启动整个推理
用于推理：InternNav/scripts/eval/server_Gr00t.py 完成构造输入，调用模型启动推理，返回结果action。更换模型时照着这个文件的内容仿写一个，换为自己的逻辑即可
用于“HTTP 插头”：InternNav/internnav/evaluator/HTTPTrajectoryClient.py 此文件只用于继承一个BaseTrajectoryClient类，是HTTP Trajectory Server 的 Client，更换模型时按照模型需要使用正确方式包裹发送即可
基本类即函数定义：InternNav/internnav/evaluator/final_habitat_vln_evaluator.py 更换模型后如果有新的参数或逻辑，可以在这里添补

## Prepared ScaleVLN routes

`scripts/eval/configs/vln_scalevln_no_oracle.yaml` selects the native Habitat
`ScaleVLNVLN-v1` dataset. Set `SCALEVLN_ROOT` to the ScaleVLN dataset root; the YAML
reads `assets/scalevln_0918_40k/selection_manifest.json` and its ordered gzip JSONL
route shards, and resolves scene paths below `assets`. Only schema 1 / train is
supported. Each instruction becomes a standard `VLNEpisode`; `positions` supplies
the full reference path, including start/goal. Source Habitat XYZ coordinates and
XYZW rotations are preserved. No processed videos or offline conversion are needed.

The supplied manifest has 40,000 episodes in 800 scenes; the 39,974 processed
episodes are a separate collection. R2R/RxR keep `R2RVLN-v1` and their existing
parser. ScaleVLN uses the same evaluator, four-view sensors and replay teacher.
The YAML camera is 640×480, HFOV 120°, height 1.5m. For horizontal views, set
`INIT_LOOK_DOWN_STEPS=0`; the Enactive launcher otherwise looks down twice.
Set success distance explicitly
in the launcher: the Enactive pool defaults to 1m and overrides the YAML; for 3m
collection use `enactive dagger collect ... --success-distance 3.0`.
CPU dataset/catalog tests do not establish renderer or closed-loop teacher health.

使用方法：
1.首先运行：
uvicorn InternNav.scripts.eval.server_Gr00t:app \
    --host 127.0.0.1 \
    --port 9000
2.接着运行：
python scripts/eval/eval_main.py --model_path /data/sjh/GR00T-Internva/output_uav/checkpoint-300000 --continuous_traj --output_path result/Gr00t/val_unseen_32traj_8steps --save-snapshots

## Enactive HTTP action contract

The Habitat HTTP client accepts legacy `actions` responses and Enactive canonical
schema-v2 `continuous_action[16][4]` responses. For canonical chunks, the client executes only the
`chunk_execute_horizon` prefix, reconstructs it with `canonical_relative_v1` SE(2)
semantics, and performs Habitat discretization locally. Observations advertise
`client_capabilities` and report the previous query's actually executed discrete
actions in `executed_actions`. STOP, oracle-goal following, and transparent replan
ACK behavior remain client-side compatible. The capability declaration includes
`high_policy_replan_ack_v1`; protocol errors terminate the rollout rather than being
counted as a normal STOP.
