"""Load prepared ScaleVLN route shards as native VLN episodes."""

import gzip
import hashlib
import json
import math
from pathlib import Path

from habitat.core.registry import registry
from habitat.datasets.utils import VocabDict
from habitat.datasets.vln.r2r_vln_dataset import VLNDatasetV1
from habitat.tasks.nav.nav import NavigationGoal
from habitat.tasks.vln.vln import InstructionData, VLNEpisode


def _vector(value, size, field):
    if (
        not isinstance(value, list)
        or len(value) != size
        or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)
    ):
        raise ValueError(f"{field} must contain {size} finite numbers")
    return value


def _count(value):
    if type(value) is not int or value < 0:
        raise ValueError("manifest counts must be nonnegative integers")
    return value


def _route_episodes(route, scenes_dir):
    if route["schema_version"] != 1:
        raise ValueError("unsupported route schema_version")
    scene = route["scene_id"]
    if not isinstance(scene, str) or not scene or Path(scene).is_absolute() or ".." in Path(scene).parts:
        raise ValueError("scene_id must be relative to scenes_dir")
    start = _vector(route["start_position"], 3, "start_position")
    rotation = _vector(route["start_rotation"], 4, "start_rotation")
    if not math.isclose(sum(v * v for v in rotation), 1.0, abs_tol=1e-4):
        raise ValueError("start_rotation must be a unit XYZW quaternion")
    goal = _vector(route["goal_position"], 3, "goal_position")
    radius = route["goal_radius"]
    if type(radius) not in (int, float) or not math.isfinite(radius) or radius <= 0:
        raise ValueError("goal_radius must be finite and positive")
    positions = route["positions"]
    if not isinstance(positions, list) or len(positions) < 2:
        raise ValueError("positions must include start and goal")
    for point in positions:
        _vector(point, 3, "positions")
    for point, expected in ((positions[0], start), (positions[-1], goal)):
        if any(not math.isclose(a, b, rel_tol=0, abs_tol=1e-5) for a, b in zip(point, expected)):
            raise ValueError("positions endpoints must match start_position and goal_position")
    instructions, ids = route["instructions"], route["instr_ids"]
    if (
        not isinstance(instructions, list)
        or not instructions
        or not isinstance(ids, list)
        or len(ids) != len(instructions)
        or any(not isinstance(v, str) or not v.strip() for v in instructions + ids)
    ):
        raise ValueError("instructions and instr_ids must be matching nonempty string lists")
    trajectory_id = route["trajectory_id"]
    if not isinstance(trajectory_id, (str, int)) or isinstance(trajectory_id, bool) or trajectory_id == "":
        raise ValueError("trajectory_id must be a nonempty string or integer")
    return [
        VLNEpisode(
            episode_id=episode_id,
            trajectory_id=trajectory_id,
            scene_id=str(Path(scenes_dir) / scene),
            start_position=list(start),
            start_rotation=list(rotation),
            instruction=InstructionData(instruction_text=instruction),
            goals=[NavigationGoal(position=list(goal), radius=radius)],
            # The raw reference_path omits the two endpoints.
            reference_path=[list(point) for point in positions],
        )
        for episode_id, instruction in zip(ids, instructions)
    ]


@registry.register_dataset(name="ScaleVLNVLN-v1")
class ScaleVLNDatasetV1(VLNDatasetV1):
    """Schema-1 selection_manifest.json plus ordered gzip JSONL route shards.

    Uses the shared VLNEpisode contract; coordinates are already Habitat XYZ,
    and start_rotation is XYZW. No simulator or offline conversion is needed.
    """

    def __init__(self, config=None):
        super().__init__()
        self.instruction_vocab = VocabDict(word_list=[])
        self._catalog_metadata = None
        if config is None:
            return
        if config.split != "train":
            raise ValueError("ScaleVLNVLN-v1 only supports the prepared train split")
        manifest_path = Path(config.data_path)
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest["schema_version"] != 1:
            raise ValueError("unsupported ScaleVLN manifest schema_version")
        if manifest["content_pattern"] != "shard_{shard}/routes.jsonl.gz":
            raise ValueError("unsupported ScaleVLN content_pattern")
        route_counts = manifest["shard_route_counts"]
        instruction_counts = manifest["shard_instruction_counts"]
        if (
            not isinstance(route_counts, list)
            or not route_counts
            or not isinstance(instruction_counts, list)
            or len(route_counts) != len(instruction_counts)
        ):
            raise ValueError("manifest shard counts must be matching nonempty lists")
        for value in route_counts + instruction_counts:
            _count(value)
        if sum(route_counts) != _count(manifest["selected_route_count"]):
            raise ValueError("manifest selected_route_count mismatch")
        if sum(instruction_counts) != _count(manifest["selected_instruction_count"]):
            raise ValueError("manifest selected_instruction_count mismatch")
        digest = hashlib.sha256()
        ids, scenes = set(), set()
        for shard, expected_routes in enumerate(route_counts):
            path = manifest_path.parent / manifest["content_pattern"].format(shard=shard)
            num_routes = num_instructions = 0
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    try:
                        route = json.loads(line)
                        episodes = _route_episodes(route, config.scenes_dir)
                        for episode in episodes:
                            if episode.episode_id in ids:
                                raise ValueError(f"duplicate episode ID: {episode.episode_id}")
                            ids.add(episode.episode_id)
                        # Hash source content before scene filtering, without deployment paths.
                        digest.update(
                            json.dumps(route, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
                        )
                        digest.update(b"\n")
                        scenes.add(route["scene_id"])
                        self.episodes.extend(episodes)
                        num_routes += 1
                        num_instructions += len(episodes)
                    except (KeyError, TypeError, ValueError) as exc:
                        raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if (num_routes, num_instructions) != (expected_routes, instruction_counts[shard]):
                raise ValueError(f"{path}: route/instruction counts do not match manifest")
        if len(scenes) != _count(manifest["selected_scene_count"]):
            raise ValueError("manifest selected_scene_count mismatch")
        self._catalog_metadata = {
            "dataset_type": "ScaleVLNVLN-v1",
            "adapter_schema_version": 1,
            "routes_sha256": digest.hexdigest(),
            "route_count": sum(route_counts),
            "instruction_count": sum(instruction_counts),
            "scene_count": len(scenes),
        }
        self.episodes = list(filter(self.build_content_scenes_filter(config), self.episodes))

    def get_catalog_metadata(self):
        """Return the source identity shared by all scene-filtered workers."""
        if self._catalog_metadata is None:
            raise ValueError("ScaleVLN catalog metadata requires a manifest-backed dataset")
        return dict(self._catalog_metadata)
