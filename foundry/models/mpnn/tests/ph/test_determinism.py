"""run_ph_redesign returns the same designs whatever n_jobs is, and drops none silently."""

import time

from engine_fixtures import bare_engine, make_ctx, make_output


def test_run_ph_redesign_returns_the_same_designs_for_any_n_jobs(engine, monkeypatch):
    def fake_task(eng, ctx, criteria_list, base_seed, task):
        index, _ = task
        time.sleep(
            0.06 * (3 - index)
        )  # later tasks finish first: arrival != task order
        variant = ["ALA", "HIS", "ALA"] if index % 2 else ["ALA", "ALA", "GLY"]
        return engine.PHDesignSet(
            [
                make_output(
                    engine, 0, -1.0 - index, ["ALA"] * 3, index
                ),  # repeats exactly
                make_output(engine, 1, -2.0, variant, index),  # same id, two sequences
            ]
        )

    monkeypatch.setattr(engine, "_run_design_task", fake_task)
    runner = bare_engine(engine, make_ctx(engine))
    criteria = [
        engine.PHDesignCriteria(
            method="block_descent", seed_source="native", combined_lambda=lam
        )
        for lam in (0.0, 0.3, 0.6, 1.0)
    ]

    def run(n_jobs):
        designs = runner.run_ph_redesign(
            atom_array=None,
            binder_chain="A",
            criteria_list=criteria,
            seed=0,
            n_jobs=n_jobs,
        )
        return [
            (d.design_id(), d.seed_idx, tuple(d.extended_tokens), d.final_potts_energy)
            for d in designs
        ]

    serial, parallel = run(1), run(3)
    assert serial == parallel
    ids = [row[0] for row in serial]
    assert len(ids) == len(set(ids)) == 3  # exact repeats dropped, both sequences kept
    assert sorted(i for i in ids if "~" in i) == ["s_s1~2"]
    assert {row[1] for row in serial} == {0, 1}  # first task wins each repeat


def test_deduped_keeps_distinct_sequences_that_share_an_id(engine):
    a = make_output(engine, 0, -1.0, ["ALA", "GLY"], 0)
    b = make_output(engine, 0, -1.5, ["ALA", "HIS"], 0)
    again = make_output(engine, 0, -1.2, ["ALA", "GLY"], 0)
    kept = engine.PHDesignSet([a, b, again]).deduped()
    assert [d.design_id() for d in kept] == ["s_s0", "s_s0~2"]
