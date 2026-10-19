"""VLN-CE nDTW over native simulator positions and dense GT locations."""

from dataclasses import dataclass

import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from habitat.config.default_structured_configs import MeasurementConfig
from habitat.core.embodied_task import Measure
from habitat.core.registry import registry
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from enactive.eval.online.habitat_ndtw import (
    SCALE_M, ground_truth_locations, load_ground_truth, locations,
)


@dataclass
class NDTWMeasurementConfig(MeasurementConfig):
    type: str = "NDTW"
    gt_path: str = MISSING


ConfigStore.instance().store(
    group="habitat/task/measurements", name="ndtw",
    node=NDTWMeasurementConfig, package="habitat.task.measurements.ndtw",
)


@registry.register_measure
class NDTW(Measure):
    cls_uuid = "ndtw"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._gt, self.contract = load_ground_truth(config.gt_path)
        super().__init__(**kwargs)

    def _get_uuid(self, *args, **kwargs):
        return self.cls_uuid

    def reset_metric(self, episode, *args, **kwargs):
        self._reference = ground_truth_locations(self._gt, episode.episode_id)
        self._agent_path = []
        self.update_metric(episode=episode, *args, **kwargs)

    def update_metric(self, episode, *args, **kwargs):
        position = locations([self._sim.get_agent_state().position.tolist()])[0]
        if self._agent_path and position == self._agent_path[-1]:
            return
        self._agent_path.append(position)
        distance, _ = fastdtw(self._agent_path, self._reference, radius=1, dist=euclidean)
        self._metric = float(np.exp(-distance / (len(self._reference) * SCALE_M)))
        if not np.isfinite(self._metric):
            raise ValueError("non-finite nDTW")
