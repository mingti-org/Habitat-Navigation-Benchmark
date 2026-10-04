"""Dynamic task source for the existing VLN evaluator, without a second action loop."""

from __future__ import annotations

import gc
import json
import math
import os
import random
import time
from pathlib import Path

from enactive.eval.online.episode_scheduler_protocol import SchedulerClient, atomic_json


def run_scheduled(evaluator):
    import habitat
    import numpy as np

    scheduler = SchedulerClient.from_environment()
    if scheduler is None:
        raise ValueError("scheduled evaluator requires a coordinator")
    catalog = json.loads(Path(os.environ["ENACTIVE_EVALUATION_CATALOG"]).read_text())
    episodes = {(str(ep.scene_id).split("/")[-2], str(ep.episode_id)): ep
                for ep in evaluator._scheduled_dataset.episodes}
    if len(episodes) != len(evaluator._scheduled_dataset.episodes):
        raise ValueError("duplicate native scene/episode identity")
    current_scene = None
    collection = os.environ.get("ENACTIVE_DAGGER_COLLECT") == "1"
    def close_environment():
        nonlocal current_scene
        if evaluator.env is not None:
            environment = evaluator.env
            evaluator.env = None
            try:
                evaluator.agent.set_env(None)
            finally:
                try:
                    environment.close()
                finally:
                    gc.collect()
        current_scene = None
    try:
        while True:
            response = scheduler.claim(scene_path=current_scene) if collection else scheduler.claim()
            if response.get("stop"):
                return
            task = response.get("task")
            if task is None:
                time.sleep(response["wait_seconds"])
                continue
            scheduler.start(task)
            try:
                payload = task["payload"]
                episode = episodes[(payload["scene_id"], payload["episode_id"])]
                if str(episode.scene_id) != payload["scene_path"] or catalog["backend"] != "habitat":
                    scheduler.error(task, "native catalog identity mismatch", invalid=True)
                    raise ValueError("native catalog identity mismatch")
                if current_scene != str(episode.scene_id):
                    close_environment()
                random.seed(task["seed"])
                np.random.seed(task["seed"])
                if evaluator.env is None:
                    import copy
                    dataset = copy.copy(evaluator._scheduled_dataset)
                    dataset.episodes = [episode]
                    with habitat.config.read_write(evaluator.config):
                        evaluator.config.habitat.seed = task["seed"]
                        evaluator.config.habitat.simulator.seed = task["seed"]
                    evaluator.env = habitat.Env(config=evaluator.config, dataset=dataset)
                    evaluator.agent.set_env(evaluator.env)
                    current_scene = str(episode.scene_id)
                # Reset after construction as well: warm and fresh environments
                # must start the episode with the same Python/NumPy RNG streams.
                random.seed(task["seed"])
                np.random.seed(task["seed"])
                evaluator.env.seed(task["seed"])
                evaluator.output_path = str(Path(task["result_path"]).parent)
                Path(evaluator.output_path).mkdir(parents=True, exist_ok=True)
                client = evaluator.agent.traj_client
                client._scheduler_task = task
                client._client_session_id = task["lease_id"]
                if collection:
                    client.debug_output_path = str(Path(evaluator.output_path) / "enactive_server_snapshots")
                evaluator.run_episode(episode)
                row = evaluator._last_result_record
                if not isinstance(row, dict):
                    raise RuntimeError("native episode produced no result")
                if collection:
                    row["server_episode_end"] = client.episode_end_result
                    if row.get("termination_kind") == "accepted_teacher_unavailable":
                        # An unreachable target has no finite geodesic distance.
                        # Keep the terminal reason and receipt, using JSON null
                        # for undefined distances instead of retrying the episode.
                        for key in ("ne", "final_distance_to_goal", "min_distance", "shortest_path_length"):
                            if not math.isfinite(float(row[key])):
                                row[key] = None
                # Main-environment validator runs at the coordinator. Native code
                # must not import heavyweight/3.10-only supervisor dependencies.
                atomic_json(Path(task["result_path"]), row)
                scheduler.complete(task, row)
            except Exception as exc:
                scheduler.error(task, exc)
                raise
    finally:
        close_environment()
