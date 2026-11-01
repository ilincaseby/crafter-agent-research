"""Evaluation-only evidence collection; candidate events are not proof of intent."""

from collections import Counter
import gzip
import json

import crafter
import imageio.v2 as imageio
import numpy as np

EVENTS = {
    "connected_mining": "At least three connected resource tiles mined (not proof of a tunnel system)",
    "water_stone_placed": "Stone placed on water (not yet a crossing)",
    "water_stone_mined": "Player mined a stone they placed on water into a walkable path",
    "water_bridge_tile_traversed": "Player traversed a path made by placing and mining stone on water",
    "arrow_avoided_candidate": "Player left an arrow's next tile sideways without losing health",
    "placed_stone_blocked_arrow": "Arrow disappeared at a player-placed stone",
    "mining_before_skeleton_defeat": "Skeleton defeated within 50 steps of nearby mining (not proof of flanking)",
}


def snapshot(game):
    """Private state is read only for analysis, never provided to the policy."""
    player, world = game._player, game._world
    objects = []
    for obj in world.objects:
        if type(obj).__name__ not in ("Arrow", "Skeleton"):
            continue
        entry = {
            "id": int(world._obj_map[tuple(obj.pos)]),
            "kind": type(obj).__name__,
            "pos": [int(value) for value in obj.pos],
            "health": int(obj.health),
        }
        if entry["kind"] == "Arrow":
            entry["facing"] = [int(value) for value in obj.facing]
        objects.append(entry)
    return {
        "pos": [int(value) for value in player.pos],
        "facing": [int(value) for value in player.facing],
        "inventory": {key: int(value) for key, value in player.inventory.items()},
        "achievements": {key: int(value) for key, value in player.achievements.items()},
        "objects": objects,
    }


def component_size(cells, start):
    visited, pending = set(), [start]
    while pending:
        cell = pending.pop()
        if cell in visited or cell not in cells:
            continue
        visited.add(cell)
        x, y = cell
        pending.extend(((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)))
    return len(visited)


class BehaviorRecorder:
    def __init__(self, directory, game, observation, *, record_video=False):
        self.directory = directory
        self.game = game
        self.previous = snapshot(game)
        self.terrain = game._world._mat_map.copy()
        self.materials = game._world._mat_names.copy()
        self.step = 0
        self.counts = Counter()
        self.mined = set()
        self.placed_stones = set()
        self.water_stones = set()
        self.bridge_paths = set()
        self.traversed_bridge_paths = set()
        self.last_skeleton_mining = None
        self.max_mining_component = 0
        self.arrow_threat_steps = 0
        self.skeleton_nearby_steps = 0
        self.events = []
        self.trace = gzip.open(
            directory.with_suffix(".jsonl.gz"), "wt", encoding="utf-8"
        )
        self.video = (
            imageio.get_writer(
                directory.with_suffix(".gif"), mode="I", duration=200, loop=0
            )
            if record_video
            else None
        )
        self.write(
            {
                "type": "initial",
                "state": self.previous,
                "terrain_xy": self.terrain.tolist(),
                "material_names": self.materials,
                "video": "Existing observation, every 2 steps, through step 2000; no extra render calls",
            }
        )
        if self.video is not None:
            self.video.append_data(observation.transpose(1, 2, 0))

    def write(self, entry):
        self.trace.write(json.dumps(entry, separators=(",", ":")) + "\n")

    def observe(self, action, reward, observation):
        self.step += 1
        before, after = self.previous, snapshot(self.game)
        terrain = self.game._world._mat_map.copy()
        changes = [
            {
                "pos": [int(x), int(y)],
                "before": self.materials[int(self.terrain[x, y])],
                "after": self.materials[int(terrain[x, y])],
            }
            for x, y in np.argwhere(terrain != self.terrain)
        ]
        events = []
        action_name = crafter.constants.actions[action]
        target = tuple(a + b for a, b in zip(before["pos"], before["facing"]))
        skeleton_nearby = any(
            obj["kind"] == "Skeleton"
            and sum(abs(a - b) for a, b in zip(obj["pos"], before["pos"])) <= 8
            for obj in before["objects"]
        )
        self.skeleton_nearby_steps += int(skeleton_nearby)
        for change in changes:
            cell = tuple(change["pos"])
            was_placed = cell in self.placed_stones
            if change["after"] != "path":
                self.bridge_paths.discard(cell)
            if (
                cell in self.water_stones
                and change["after"] == "path"
                and action_name == "do"
                and cell == target
            ):
                self.bridge_paths.add(cell)
                events.append("water_stone_mined")
            if change["after"] != "stone":
                self.placed_stones.discard(cell)
                self.water_stones.discard(cell)
            if (
                action_name == "do"
                and cell == target
                and not was_placed
                and change["before"] in ("stone", "coal", "iron", "diamond")
                and change["after"] == "path"
            ):
                self.mined.add(cell)
                size = component_size(self.mined, cell)
                if size >= 3 and size > self.max_mining_component:
                    events.append("connected_mining")
                self.max_mining_component = max(self.max_mining_component, size)
                if skeleton_nearby:
                    self.last_skeleton_mining = self.step
            if (
                action_name == "place_stone"
                and cell == target
                and change["after"] == "stone"
            ):
                self.placed_stones.add(cell)
                if change["before"] == "water":
                    self.water_stones.add(cell)
                    events.append("water_stone_placed")
        position = tuple(after["pos"])
        if (
            position != tuple(before["pos"])
            and position in self.bridge_paths
            and position not in self.traversed_bridge_paths
        ):
            self.traversed_bridge_paths.add(position)
            events.append("water_bridge_tile_traversed")
        delta = np.array(after["pos"]) - before["pos"]
        after_objects = {obj["id"]: obj for obj in after["objects"]}
        for arrow in before["objects"]:
            if arrow["kind"] != "Arrow":
                continue
            next_tile = tuple(a + b for a, b in zip(arrow["pos"], arrow["facing"]))
            if next_tile == tuple(before["pos"]):
                self.arrow_threat_steps += 1
                if (
                    np.any(delta)
                    and np.dot(delta, arrow["facing"]) == 0
                    and after["inventory"]["health"] >= before["inventory"]["health"]
                    and arrow["id"] in after_objects
                    and tuple(after_objects[arrow["id"]]["pos"]) == next_tile
                ):
                    events.append("arrow_avoided_candidate")
            if (
                next_tile in self.placed_stones
                and arrow["id"] not in after_objects
                and self.game._world[next_tile][0] == "stone"
            ):
                events.append("placed_stone_blocked_arrow")
        if (
            after["achievements"]["defeat_skeleton"]
            > before["achievements"]["defeat_skeleton"]
            and self.last_skeleton_mining is not None
            and self.step - self.last_skeleton_mining <= 50
        ):
            events.append("mining_before_skeleton_defeat")
        self.counts.update(events)
        self.events.extend({"step": self.step, "event": event} for event in events)
        self.write(
            {
                "type": "step",
                "step": self.step,
                "action": action_name,
                "reward": float(reward),
                "state": after,
                "terrain_changes": changes,
                "candidates": events,
            }
        )
        if self.video is not None and self.step <= 2000 and self.step % 2 == 0:
            self.video.append_data(observation.transpose(1, 2, 0))
        self.previous, self.terrain = after, terrain

    def close(self):
        try:
            if self.video is not None:
                self.video.close()
        finally:
            self.trace.close()

    def summary(self):
        return {
            "candidate_counts": {name: self.counts[name] for name in EVENTS},
            "candidate_steps": self.events,
            "arrow_threat_steps": self.arrow_threat_steps,
            "skeleton_nearby_steps": self.skeleton_nearby_steps,
            "mined_tiles": len(self.mined),
            "largest_mined_component": self.max_mining_component,
            "video_through_step": (
                min(self.step, 2000) if self.video is not None else None
            ),
        }
