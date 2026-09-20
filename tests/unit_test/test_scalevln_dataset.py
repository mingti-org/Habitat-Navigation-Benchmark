"""CPU-only native dataset tests; no scene assets or simulator required."""

import copy
import gzip
import json
import shutil

import pytest
from omegaconf import OmegaConf

from habitat.datasets import make_dataset


def route(name="scene-a", ids=None):
    return {
        "schema_version": 1,
        "scene_id": f"hm3d/{name}/{name}.basis.glb",
        "trajectory_id": f"{name}__fake_00001",
        "instr_ids": ids or [f"{name}-instruction-1"],
        "instructions": ["Walk forward."] * len(ids or [1]),
        "start_position": [1.0, -2.5, 3.0],
        "start_rotation": [0.0, -1.0, 0.0, 0.0],
        "positions": [[1.0, -2.5, 3.0], [2.0, -2.5, 3.0], [3.0, -2.5, 3.0]],
        "reference_path": [[2.0, -2.5, 3.0]],
        "goal_position": [3.0, -2.5, 3.0],
        "goal_radius": 0.5,
    }


def write_source(root, shards=None, manifest_updates=None):
    shards = shards or [[route(ids=["a-1", "a-2"])], [route("scene-b")]]
    root.mkdir(parents=True, exist_ok=True)
    for i, rows in enumerate(shards):
        path = root / f"shard_{i}" / "routes.jsonl.gz"
        path.parent.mkdir(exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    manifest = {
        "schema_version": 1,
        "content_pattern": "shard_{shard}/routes.jsonl.gz",
        "shard_route_counts": [len(rows) for rows in shards],
        "shard_instruction_counts": [sum(len(r["instr_ids"]) for r in rows) for rows in shards],
        "selected_route_count": sum(len(rows) for rows in shards),
        "selected_instruction_count": sum(len(r["instr_ids"]) for rows in shards for r in rows),
        "selected_scene_count": len({r["scene_id"] for rows in shards for r in rows}),
    }
    manifest.update(manifest_updates or {})
    path = root / "selection_manifest.json"
    path.write_text(json.dumps(manifest))
    return OmegaConf.create(
        {"data_path": str(path), "scenes_dir": str(root / "assets"), "split": "train", "content_scenes": ["*"]}
    )


def load(config):
    return make_dataset("ScaleVLNVLN-v1", config=config)


def test_native_registration_expansion_and_geometry(tmp_path):
    config = write_source(tmp_path)
    dataset = load(config)
    assert [e.episode_id for e in dataset.episodes] == ["a-1", "a-2", "scene-b-instruction-1"]
    episode = dataset.episodes[0]
    source = route()
    assert episode.trajectory_id == source["trajectory_id"]
    assert episode.start_position == source["start_position"]
    assert episode.start_rotation == source["start_rotation"]
    assert episode.reference_path == source["positions"]
    assert episode.scene_id == str(tmp_path / "assets" / source["scene_id"])
    assert episode.goals[0].position == source["goal_position"]
    assert episode.goals[0].radius == 0.5
    assert episode.instruction.instruction_text == "Walk forward."
    assert episode.instruction.instruction_tokens is None
    assert dataset.instruction_vocab.word_list == ["<unk>"]
    assert dataset.get_catalog_metadata()["instruction_count"] == 3
    config.content_scenes = ["scene-b.basis"]
    filtered = load(config)
    assert [e.episode_id for e in filtered.episodes] == ["scene-b-instruction-1"]
    assert filtered.get_catalog_metadata() == dataset.get_catalog_metadata()


@pytest.mark.parametrize("change", ["instruction", "coordinates", "order", "id", "extra_field"])
def test_identity_binds_both_shards_and_route_order(tmp_path, change):
    shards = [[route("scene-a")], [route("scene-b")]]
    baseline = load(write_source(tmp_path, shards)).get_catalog_metadata()
    if change == "instruction":
        shards[1][0]["instructions"][0] = "Turn left."
    elif change == "coordinates":
        shards[1][0]["positions"][1][0] += 0.25
    elif change == "order":
        shards.reverse()
    elif change == "id":
        shards[1][0]["instr_ids"][0] = "another-id"
    else:
        shards[1][0]["heading"] = 0.25
    assert load(write_source(tmp_path, shards)).get_catalog_metadata() != baseline


def test_identity_is_independent_of_deployment_and_gzip_bytes(tmp_path):
    config = write_source(tmp_path / "original")
    baseline = load(config).get_catalog_metadata()
    shutil.copytree(tmp_path / "original", tmp_path / "moved")
    config.data_path = str(tmp_path / "moved" / "selection_manifest.json")
    config.scenes_dir = str(tmp_path / "new-scenes-root")
    path = tmp_path / "moved" / "shard_1" / "routes.jsonl.gz"
    with gzip.open(path, "rt") as handle:
        row = json.loads(handle.readline())
    with gzip.open(path, "wt", compresslevel=1) as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    assert load(config).get_catalog_metadata() == baseline


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("instructions", [], "matching nonempty"),
        ("instr_ids", ["same", "same"], "matching nonempty"),
        ("start_position", [0, 0], "finite numbers"),
        ("goal_position", [float("nan"), 0, 0], "finite numbers"),
        ("start_rotation", [0, 0, 0, 0], "unit XYZW"),
        ("start_rotation", [0, 0, 0, 2], "unit XYZW"),
        ("positions", [[1, -2.5, 3], [8, -2.5, 3]], "endpoints"),
        ("positions", [], "include start and goal"),
        ("goal_radius", -1, "finite and positive"),
        ("schema_version", 2, "schema_version"),
        ("scene_id", "/absolute.glb", "relative"),
    ],
)
def test_invalid_routes_fail_with_source_line(tmp_path, field, value, match):
    row = route()
    row[field] = value
    config = write_source(tmp_path, [[row]])
    with pytest.raises(ValueError, match=match) as error:
        load(config)
    assert "routes.jsonl.gz:1:" in str(error.value)


def test_missing_field_and_duplicate_ids_are_rejected(tmp_path):
    row = route()
    del row["start_rotation"]
    with pytest.raises(ValueError, match="start_rotation"):
        load(write_source(tmp_path, [[row]]))
    row = route()
    with pytest.raises(ValueError, match="duplicate episode ID"):
        load(write_source(tmp_path, [[row], [copy.deepcopy(row)]]))


@pytest.mark.parametrize(
    "updates,match",
    [
        ({"schema_version": 2}, "schema_version"),
        ({"content_pattern": "other.gz"}, "content_pattern"),
        ({"shard_instruction_counts": [2]}, "matching nonempty"),
        ({"selected_route_count": 4}, "selected_route_count"),
        ({"selected_instruction_count": 4}, "selected_instruction_count"),
        ({"shard_route_counts": [0, 2]}, "counts do not match"),
        ({"shard_instruction_counts": [1, 2]}, "counts do not match"),
        ({"selected_scene_count": 3}, "selected_scene_count"),
    ],
)
def test_manifest_contract(tmp_path, updates, match):
    with pytest.raises(ValueError, match=match):
        load(write_source(tmp_path, manifest_updates=updates))


def test_missing_shard_and_non_train_split_fail(tmp_path):
    config = write_source(tmp_path)
    (tmp_path / "shard_1" / "routes.jsonl.gz").unlink()
    with pytest.raises(FileNotFoundError):
        load(config)
    config.split = "val_unseen"
    with pytest.raises(ValueError, match="train split"):
        load(config)
