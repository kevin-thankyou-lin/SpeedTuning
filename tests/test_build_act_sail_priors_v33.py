from scripts.build_act_sail_priors_v33 import ROOTS


def test_prior_sources_are_task_specific_amlfs04_training_roots():
    assert set(ROOTS) == {"pick", "tea", "insertion"}
    assert len(set(map(str, ROOTS.values()))) == 3
    assert all(str(path).startswith("/mnt/amlfs-04/home/linke/speedtuning-original-act/") for path in ROOTS.values())
