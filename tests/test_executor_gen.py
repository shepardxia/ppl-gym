"""Gen executor (eval.executor_gen): registration + wire contract + opt-in run.

The execution test needs Julia + Gen.jl (box only); gate it behind
PPL_GYM_RUN_GEN=1 like the Stan test's PPL_GYM_RUN_STAN.
"""

import os

import pytest

from eval.corpus import BATCH_EXECUTORS
from eval.executor_gen import SERIALIZER_HEADER, _program, execute_gen_batch
from eval.gt_cache import EXECUTOR_VERSION


def test_gen_registered():
    assert BATCH_EXECUTORS.get("gen") is execute_gen_batch
    assert EXECUTOR_VERSION.get("gen")  # a version string exists


def test_program_structure():
    # The injected preamble + user code + the ANSWER emit line, in order.
    prog = _program("ANSWER = 1", seed=7)
    assert "using Gen, JSON, Random" in prog
    assert "enum_dist" in SERIALIZER_HEADER  # the enum-dist helper is injected
    assert "Random.seed!(7)" in prog
    assert prog.index("using Gen") < prog.index("Random.seed!(7)")  # header before seed
    assert prog.rstrip().endswith("println(JSON.json(_serialize_answer(ANSWER)))")


def test_empty_seeds():
    assert execute_gen_batch("ANSWER = 1", [], timeout=10) == ([], [])


@pytest.mark.skipif(os.environ.get("PPL_GYM_RUN_GEN") != "1",
                    reason="set PPL_GYM_RUN_GEN=1 (+ PPL_GYM_JULIA) for the Gen exec test")
def test_coin_matches_webppl_gt():
    # The fair/biased coin (probmods2-conditioning/ex1.c): exact posterior.
    code = r'''
@gen function coin()
    is_fair = ({:coin} ~ bernoulli(0.5))
    p = is_fair ? 0.5 : 0.9
    {:f1} ~ bernoulli(p); {:f2} ~ bernoulli(p); {:f3} ~ bernoulli(p)
    return is_fair ? "fair" : "biased"
end
obs = choicemap((:f1, true), (:f2, true), (:f3, true))
grid = choice_vol_grid((:coin, [false, true]))
ANSWER = enum_dist(enumerative_inference(coin, (), obs, grid))
'''
    answers, errors = execute_gen_batch(code, [42, 43], timeout=180)
    assert errors == [None, None]
    a = answers[0]
    m = dict(zip(a["support"], a["probs"]))
    assert abs(m["biased"] - 0.8536299766) < 1e-6
    assert abs(m["fair"] - 0.1463700234) < 1e-6
