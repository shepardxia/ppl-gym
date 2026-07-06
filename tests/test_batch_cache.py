"""Batched pyro execution + persistent GT cache (eval.executor_pyro, eval.gt_cache)."""

import os

from eval.executor_pyro import execute_pyro, execute_pyro_batch


def test_batch_determinism_same_seed():
    code = "ANSWER = [pyro.sample(f'x{i}', dist.Normal(0., 1.)).item() for i in range(3)]"
    a, _ = execute_pyro_batch(code, [7, 7], timeout=60)
    assert a[0] == a[1]  # same seed → identical


def test_batch_different_seeds_differ():
    code = "ANSWER = pyro.sample('x', dist.Normal(0., 1.)).item()"
    (a, b), _ = execute_pyro_batch(code, [1, 2], timeout=60)
    assert a != b


def test_batch_matches_per_process():
    # An object-spec program: batched run for a seed == fresh-process run.
    code = "ANSWER = dist.Bernoulli(0.7)"
    batch, _ = execute_pyro_batch(code, [42, 43], timeout=60)
    for seed, got in zip((42, 43), batch):
        single = execute_pyro(code, timeout=60, random_seed=seed)
        assert single.success
        assert got == single.answer


def test_batch_per_seed_error_isolated():
    # A program that errors should yield None for that seed without killing others,
    # and surface the real per-seed reason in the aligned errors channel.
    code = "ANSWER = undefined_name"
    answers, errors = execute_pyro_batch(code, [1, 2], timeout=60)
    assert answers == [None, None]
    assert len(errors) == 2 and all(e and "undefined_name" in e for e in errors)


def test_cache_hit_equals_miss(tmp_path, monkeypatch):
    import eval.gt_cache as gc
    monkeypatch.setattr(gc, "_CACHE_DIR", tmp_path / "gtc")
    code = "ANSWER = dist.Bernoulli(0.3)"
    miss, miss_err = gc.cached_run("pyro", code, [42, 43], timeout=60, workers=1)  # writes
    hit, hit_err = gc.cached_run("pyro", code, [42, 43], timeout=60, workers=1)    # reads
    assert miss == hit
    assert miss_err == [None, None] and hit_err == [None, None]  # no failures
    assert (tmp_path / "gtc").exists()  # a cache file was written


def test_cache_disabled_by_env(tmp_path, monkeypatch):
    import eval.gt_cache as gc
    monkeypatch.setattr(gc, "_CACHE_DIR", tmp_path / "gtc")
    monkeypatch.setenv("PPL_GYM_NO_CACHE", "1")
    code = "ANSWER = dist.Bernoulli(0.3)"
    gc.cached_run("pyro", code, [42], timeout=60, workers=1)
    assert not (tmp_path / "gtc").exists()  # nothing written when disabled
