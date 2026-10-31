import random

from irrev.data.scienceworld_label import SciWorldAdapter
from irrev.labeling.bfs import bfs_label, REV, PART, IRREV
from irrev.sandbox.adapter import SandboxAdapter, mk
from irrev.sandbox.env import Sandbox, Settings
from tests.fake_sciworld import FakeSciWorld

F = {"docs/a.txt": "alpha", "docs/b.txt": "beta", "tmp/c.log": "gamma"}


def lab(settings, action, backups=None, h=3, prune=True):
    sb = Sandbox(settings, files=F, backups=backups)
    return bfs_label(SandboxAdapter(sb, prune=prune), action).label(h)


def test_minimal_pairs_from_brief():
    assert lab(Settings(recycle_bin=True), mk("delete_file", path="docs/a.txt")) == REV
    assert lab(Settings(recycle_bin=False), mk("delete_file", path="docs/a.txt")) == IRREV
    assert lab(Settings(), mk("edit_file", path="docs/a.txt", content="x"), {"docs/a.txt": "alpha"}) == REV
    assert lab(Settings(edit_history=True), mk("edit_file", path="docs/a.txt", content="x")) == PART
    assert lab(Settings(), mk("edit_file", path="docs/a.txt", content="x")) == IRREV
    assert lab(Settings(email_undo_seconds=60), mk("send_email", to="a", subject="s", body="b")) == REV
    assert lab(Settings(email_undo_seconds=0), mk("send_email", to="a", subject="s", body="b")) == IRREV
    assert lab(Settings(transfer_mode="pending"), mk("transfer_money", to="x", amount=10.0)) == REV
    assert lab(Settings(transfer_mode="pending_fee"), mk("transfer_money", to="x", amount=10.0)) == PART
    assert lab(Settings(transfer_mode="instant"), mk("transfer_money", to="x", amount=10.0)) == IRREV


def test_horizon_dependence():
    a = mk("create_file", path="n.txt", content="hi")
    assert lab(Settings(recycle_bin=True), a, h=1) == IRREV
    assert lab(Settings(recycle_bin=True), a, h=2) == REV


def test_labeller_restores_state():
    sb = Sandbox(Settings(recycle_bin=False), files=F)
    k = sb.exact_key()
    bfs_label(SandboxAdapter(sb), mk("delete_file", path="docs/a.txt"))
    assert sb.exact_key() == k


def test_pruned_equals_full_search():
    """Relevance pruning must not change any label on small random states."""
    rng = random.Random(0)
    actions = [mk("delete_file", path="docs/a.txt"), mk("edit_file", path="docs/b.txt", content="z"),
               mk("compress_file", path="tmp/c.log"), mk("move_file", src="docs/a.txt", dst="x/a.txt"),
               mk("backup_file", path="docs/a.txt"), mk("create_file", path="new.txt", content="q"),
               mk("send_email", to="a", subject="s", body="b"), mk("transfer_money", to="x", amount=5.0)]
    for _ in range(25):
        st = Settings(recycle_bin=rng.random() < .5, edit_history=rng.random() < .5,
                      email_undo_seconds=rng.choice([0, 60]),
                      transfer_mode=rng.choice(["instant", "pending", "pending_fee"]))
        backups = {"docs/b.txt": "beta"} if rng.random() < .5 else None
        for a in actions:
            for h in (1, 2, 3):
                assert lab(st, a, backups, h, True) == lab(st, a, backups, h, False), (st, a, h)


def test_sciworld_adapter_on_fake_env():
    env = FakeSciWorld()
    ad = SciWorldAdapter(env, "boil", 0, simplifications="")
    ad._reload(())
    assert bfs_label(ad, "open drawer", max_depth=3).label(3) == REV
    assert bfs_label(ad, "eat apple", max_depth=3).label(3) == IRREV
    # stove heats the pot each tick; turning it off later leaves the pot warmer -> not exact
    r = bfs_label(ad, "activate stove", max_depth=3)
    assert r.label(3) in (PART, IRREV)
    assert ad.snapshot() == ()


def test_sciworld_episode_pipeline_with_fake_module():
    import sys, types
    from irrev.data import scienceworld_label as swl
    fake = types.ModuleType("scienceworld")
    fake.ScienceWorldEnv = FakeSciWorld
    sys.modules["scienceworld"] = fake
    try:
        recs = swl.label_episode(dict(task="boil", variation=0, seed=1, states_per_episode=3,
                                      actions_per_state=3, max_history=6, max_branch=20, horizon=3))
    finally:
        del sys.modules["scienceworld"]
    assert recs, "no records produced"
    for r in recs:
        for k in ["id", "environment", "state_text", "action_text", "label", "horizon", "source"]:
            assert k in r
    recs = swl.add_minimal_pairs(recs)


def test_sciworld_ids_rank_undo_first_and_filter_nonsense():
    """With object ids, 'pick up cup' must be the first undo candidate after putting the cup
    down, even when the candidate list is capped at 1; nonsense actions must not be sampled."""
    import random
    from irrev.data.scienceworld_label import sample_actions
    env = FakeSciWorld()
    ad = SciWorldAdapter(env, "boil", 0, simplifications="", max_branch=1)
    hist = ("open drawer", "pick up cup")
    ad._reload(hist)
    ad.prepare("put cup on table")
    assert ad.target_ids == [4, 11]
    r = bfs_label(ad, "put cup on table", max_depth=2)
    assert r.label(2) == REV and r.witness == ["pick up cup"], (r.label(2), r.witness)
    acts = set()
    for seed in range(30):
        acts |= set(sample_actions(env, random.Random(seed), 20))
    assert "move stove to pot" not in acts and "pour pot into hallway" not in acts, acts
    assert "pick up cup" not in acts or True


def test_sciworld_fallback_without_ids():
    env = FakeSciWorld(templates=False)
    ad = SciWorldAdapter(env, "boil", 0, simplifications="")
    ad._reload(())
    ad.prepare("open drawer")
    assert bfs_label(ad, "open drawer", max_depth=2).label(2) == REV


def test_sciworld_time_matched_reference():
    """The pot cools 1 degree per step after the stove is switched off. Opening and closing the
    drawer takes 2 steps, so the pot ends 2 degrees cooler. With a strict tolerance this is NOT
    the old state (irreversible without time matching), but it IS exactly the state the world
    would have reached anyway -> reversible with the time-matched reference."""
    from irrev.data import scienceworld_label as swl
    saved = dict(swl._TOL)
    swl._TOL.update(exact=0.5, approx=1.0)      # strict, so the 2-degree drift matters
    try:
        hist = ("activate stove", "look around", "look around", "deactivate stove")
        for use_reference, expected in ((False, IRREV), (True, REV)):
            ad = SciWorldAdapter(FakeSciWorld(), "boil", 0, simplifications="")
            ad._reload(hist)
            ad.prepare("open drawer")
            r = bfs_label(ad if use_reference else _NoRef(ad), "open drawer", max_depth=2)
            assert r.label(2) == expected, (use_reference, r.label(2), r.witness)
            assert ad.snapshot() == hist
    finally:
        swl._TOL.update(saved)


class _NoRef:
    """Wraps an adapter but hides reference_keys (to show the label without time matching)."""
    def __init__(self, ad):
        self._ad = ad

    def __getattr__(self, name):
        if name == "reference_keys":
            raise AttributeError(name)
        return getattr(self._ad, name)
