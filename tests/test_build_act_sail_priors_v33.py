from scripts.build_act_sail_priors_v33 import OFFLINE_ARTIFACTS, ROOTS


def test_prior_sources_are_task_specific_amlfs04_training_roots():
    assert set(ROOTS) == {"pick", "tea", "insertion"}
    assert len(set(map(str, ROOTS.values()))) == 3
    assert all(str(path).startswith("/mnt/amlfs-04/home/linke/speedtuning-original-act/") for path in ROOTS.values())


def test_prior_reuses_audited_v1_offline_artifacts():
    assert set(OFFLINE_ARTIFACTS) == set(ROOTS)
    assert all(
        "speedtuning-act-speed-benchmark-v1/runs/298c6d16784f228df0b1f455d0e41b4276ec5184"
        in str(path)
        for path in OFFLINE_ARTIFACTS.values()
    )
    assert all(path.name == "offline_artifact.json" for path in OFFLINE_ARTIFACTS.values())
