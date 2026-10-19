"""Run in the native Habitat interpreter; no simulator assets required."""
import gzip
import json
from types import SimpleNamespace

import numpy as np
import pytest
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from hydra import compose, initialize
from habitat.core.embodied_task import Measurements

from internnav.evaluator.ndtw import NDTW


def test_measure_matches_official_and_copies_positions(tmp_path):
    reference = [[0., 0., 0.], [1., .5, 0.], [2., 0., 0.]]
    path = tmp_path / 'gt.json.gz'
    path.write_bytes(gzip.compress(json.dumps({'1': {'locations': reference}}).encode()))
    position = np.array([0., 0., 0.])
    sim = SimpleNamespace(get_agent_state=lambda: SimpleNamespace(position=position))
    measure = NDTW(sim, SimpleNamespace(gt_path=str(path)))
    episode = SimpleNamespace(episode_id='1')
    measure.reset_metric(episode)
    measure.update_metric(episode)  # stationary rotation / look / stop
    assert measure._agent_path == [[0., 0., 0.]]
    # Native task's Measurements dispatcher is the hook used by Env.step.
    dispatcher = Measurements([measure])
    for x in np.linspace(.25, 2., 8):
        position[0] = x
        dispatcher.update_measures(episode=episode, task=None, action=None)
    expected_path = [[0., 0., 0.]] + [[x, 0., 0.] for x in np.linspace(.25, 2., 8)]
    assert measure._agent_path == expected_path
    distance, _ = fastdtw(expected_path, reference, dist=euclidean)
    assert measure.get_metric() == pytest.approx(np.exp(-distance / (len(reference) * 3.)))
    measure.update_metric(episode)
    assert len(measure._agent_path) == 9
    with pytest.raises(ValueError, match='missing'):
        measure.reset_metric(SimpleNamespace(episode_id='absent'))
    position[0] = float('nan')
    with pytest.raises(ValueError, match='non-finite'):
        measure.reset_metric(episode)


def test_measurement_hydra_registration():
    with initialize(version_base=None):
        cfg = compose(overrides=['+habitat/task/measurements=ndtw',
                                 'habitat.task.measurements.ndtw.gt_path=gt.json.gz'])
    assert cfg.habitat.task.measurements.ndtw.type == 'NDTW'
    assert cfg.habitat.task.measurements.ndtw.gt_path == 'gt.json.gz'


def test_real_env_samples_each_native_action(tmp_path):
    import os
    import habitat
    from habitat_baselines.config.default import get_config
    from habitat.datasets import make_dataset

    config_path = os.environ.get('HABITAT_NDTW_TEST_CONFIG')
    if not config_path:
        pytest.skip('requires native scene assets and HABITAT_NDTW_TEST_CONFIG')
    scores = []
    for threshold in (1., 3.):
        config = get_config(config_path)
        with habitat.config.read_write(config):
            config.habitat.task.measurements.success.success_distance = threshold
            config.habitat.simulator.agents.main_agent.sim_sensors = {}
            config.habitat.simulator.create_renderer = False
        dataset = make_dataset(config.habitat.dataset.type, config=config.habitat.dataset)
        dataset.episodes = dataset.episodes[:1]
        with habitat.Env(config=config, dataset=dataset) as env:
            env.reset()
            measure = env.task.measurements.measures['ndtw']
            actual = [env.sim.get_agent_state().position.tolist()]
            for _ in range(8):
                env.step(1)
                pos = env.sim.get_agent_state().position.tolist()
                if pos != actual[-1]:
                    actual.append(pos)
                assert measure._agent_path == actual
            env.step(2)  # pure rotation adds no point
            assert measure._agent_path == actual
            distance, _ = fastdtw(actual, measure._reference, dist=euclidean)
            assert env.get_metrics()['ndtw'] == pytest.approx(np.exp(-distance / (3 * len(measure._reference))))
            scores.append(env.get_metrics()['ndtw'])
    assert scores[0] == scores[1]
