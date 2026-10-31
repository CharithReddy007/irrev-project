"""A tiny deterministic stand-in for ScienceWorldEnv, used only by the tests.

It mimics the parts of the API the labeller uses: load, reset, step, getObjectTree,
get_valid_action_object_combinations, look, inventory, get_task_description,
get_gold_action_sequence, close. Time passes on every step (the stove heats its contents),
like in the real simulator.
"""
import copy


class _FakeServer:
    """Mimics env.server (the Java side): raw step/reset without the extra info calls."""
    def __init__(self, env):
        self.env = env

    def reset(self):
        self.env._init()

    def step(self, a):
        return self.env.step(a)[0]


IDS = {"universe": 1, "kitchen": 2, "drawer": 3, "cup": 4, "apple": 5, "stove": 6, "pot": 7,
       "sink": 8, "agent": 9, "water": 10, "table": 11, "hallway": 12}


class FakeSciWorld:
    """templates=True mimics the real API (object ids per valid action, realistic object tree);
    templates=False removes that method, to test the name-matching fallback."""

    def __init__(self, *a, templates=True, **k):
        self.loaded = False
        self.server = _FakeServer(self)
        if not templates:
            self.get_valid_action_object_combinations_with_templates = None

    def load(self, task, var=0, simp="", generateGoldPath=False):
        self.task, self.var = task, var
        self.loaded = True

    def _init(self):
        self.w = {
            "agent_room": "kitchen",
            "drawer_open": False,
            "cup_loc": "drawer",          # drawer | inventory | table
            "apple_exists": True,
            "stove_on": False,
            "pot_temp": 20.0,
            "water_in_pot": True,
        }

    def reset(self):
        self._init()
        return self.step("look around")[0], {}

    def _tick(self):
        if self.w["stove_on"] and self.w["water_in_pot"]:
            self.w["pot_temp"] = min(100.0, self.w["pot_temp"] + 15)
        elif self.w["pot_temp"] > 20:
            self.w["pot_temp"] = max(20.0, self.w["pot_temp"] - 1)

    def get_valid_action_object_combinations(self):
        w = self.w
        acts = ["look around", "inventory", "go to hallway" if w["agent_room"] == "kitchen" else "go to kitchen"]
        if w["agent_room"] == "kitchen":
            acts.append("close drawer" if w["drawer_open"] else "open drawer")
            if w["drawer_open"] and w["cup_loc"] == "drawer":
                acts.append("pick up cup")
            if w["cup_loc"] == "inventory":
                acts += ["put cup in drawer"] if w["drawer_open"] else []
                acts.append("put cup on table")
            if w["cup_loc"] == "table":
                acts.append("pick up cup")
            if w["apple_exists"]:
                acts.append("eat apple")
            acts.append("deactivate stove" if w["stove_on"] else "activate stove")
            if w["water_in_pot"]:
                acts.append("pour pot in sink")
            acts.append("focus on cup")
            acts.append("move stove to pot")        # valid but meaningless (stove is not movable)
            acts.append("pour pot into hallway")    # valid but meaningless (target is a room)
        return acts

    def get_valid_action_object_combinations_with_templates(self):
        out = []
        for a in self.get_valid_action_object_combinations():
            words = a.replace("pick up", "pick_up").split()
            ids = [IDS[w] for w in words if w in IDS]
            out.append({"action": a, "template_id": 0, "obj_ids": ids})
        return out

    def step(self, a):
        w = self.w
        obs = "No known action matches that input."
        if a == "look around":
            obs = self.look()
        elif a in self.get_valid_action_object_combinations():
            if a.startswith("go to"):
                w["agent_room"] = a.split()[-1]
            elif a == "open drawer":
                w["drawer_open"] = True
            elif a == "close drawer":
                w["drawer_open"] = False
            elif a == "pick up cup":
                w["cup_loc"] = "inventory"
            elif a == "put cup in drawer":
                w["cup_loc"] = "drawer"
            elif a == "put cup on table":
                w["cup_loc"] = "table"
            elif a == "eat apple":
                w["apple_exists"] = False
            elif a == "activate stove":
                w["stove_on"] = True
            elif a == "deactivate stove":
                w["stove_on"] = False
            elif a == "pour pot in sink":
                w["water_in_pot"] = False
            obs = f"You {a}."
        self._tick()
        return obs, 0, False, {"score": 0}

    def getObjectTree(self):
        w = self.w

        def obj(name, movable, container, **props):
            return {"uuid": str(IDS[name]), "name": name, "type": name,
                    "propMoveable": {"isMovable": movable},
                    "propContainer": {"isContainer": "true" if container else "false", "isOpen": True,
                                      "isClosable": False} if container else None,
                    "propMaterial": {"temperatureC": props.pop("temp", 20.0)}, **props, "contents": {}}

        rooms = {"kitchen": obj("kitchen", False, True), "hallway": obj("hallway", False, True)}
        k = rooms["kitchen"]["contents"]
        drawer = obj("drawer", False, True, isOpen=w["drawer_open"]); k["drawer-3"] = drawer
        table = obj("table", False, True); k["table-11"] = table
        k["stove-6"] = obj("stove", False, False, isActivated=w["stove_on"])
        pot = obj("pot", True, True, temp=w["pot_temp"]); k["pot-7"] = pot
        if w["water_in_pot"]:
            pot["contents"]["water-10"] = obj("water", True, False, temp=w["pot_temp"])
        k["sink-8"] = obj("sink", False, True)
        if w["apple_exists"]:
            k["apple-5"] = obj("apple", True, False)
        agent = obj("agent", True, True)
        rooms[w["agent_room"]]["contents"]["agent-9"] = agent
        cup = obj("cup", True, True)
        {"drawer": drawer, "table": table, "inventory": agent}[w["cup_loc"]]["contents"]["cup-4"] = cup
        return {"uuid": "1", "name": "universe", "type": "universe", "contents":
                {f"{n}-{IDS[n]}": r for n, r in rooms.items()}}

    def get_possible_objects(self):
        return ["drawer", "cup", "table", "apple", "stove", "pot", "sink", "hallway", "kitchen"]

    def look(self):
        return f"You are in the {self.w['agent_room']}."

    def inventory(self):
        return "cup" if self.w["cup_loc"] == "inventory" else "nothing"

    def get_task_description(self):
        return "Your task is to boil water."

    def get_gold_action_sequence(self):
        return ["open drawer", "pick up cup", "put cup on table", "activate stove", "look around", "deactivate stove"]

    def close(self):
        pass
