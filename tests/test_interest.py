import numpy as np
from mmrr.interest import interest_scores, METHODS


def test_multiple_interests_and_recent_blending():
    sims = np.array([[.9,.6,.1],[.1,.6,.8]],dtype=np.float32)
    assert np.argmax(interest_scores(sims,'history_max')) == 0
    assert np.argmax(interest_scores(sims,'last')) == 2
    assert np.argmax(interest_scores(sims,'last2_mean')) == 1
    assert np.argmax(interest_scores(sims,'max_last_050')) == 2
    np.testing.assert_allclose(interest_scores(sims,'history_top2_mean'),sims.mean(axis=0))


def test_single_history_all_methods_agree_and_do_not_mutate_input():
    sims = np.array([[-.5,.2,.8]],dtype=np.float32)
    original = sims.copy()
    for method in METHODS:
        np.testing.assert_allclose(interest_scores(sims,method),sims[0])
    np.testing.assert_array_equal(sims,original)
