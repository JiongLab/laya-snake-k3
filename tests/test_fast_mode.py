"""Fast mode must keep real movement probabilities, not invent auxiliary estimates."""
import pytest
from snake_k3.game import DIRECTIONS, SnakeGame
from snake_k3.policy import LayaPolicy
from snake_k3.ui import compose


class FixedInference:
    """Inference boundary double; exercise the real policy and renderer."""
    def __init__(self):
        self.questions = None

    def predict(self, state, questions):
        self.questions = questions
        return {"answers": {"move": {"probabilities": {"UP": .1, "DOWN": .2, "LEFT": .3, "RIGHT": .4}},
                            "risk": {"noul": .9}, "food": {"noul": .8}},
                "usage": {"input_tokens": 64, "output_tokens": 0}}


def test_fast_mode_omits_auxiliary_estimates_and_only_asks_for_move():
    policy = LayaPolicy.__new__(LayaPolicy)
    policy.agent = FixedInference()
    policy.guarded, policy.prompt, policy.full_metrics = True, "compact", False
    game = SnakeGame()
    decision = policy.decide(game)
    assert decision.dead_end_risk is None and decision.food_reachable is None
    assert set(policy.agent.questions) == {"move"}
    assert decision.probabilities == {"UP": .1, "DOWN": .2, "LEFT": .3, "RIGHT": .4}
    assert decision.executed in decision.safe_directions
    assert decision.intervened == (decision.proposed != decision.executed)


def test_uncomputed_metrics_are_labeled_not_rendered_as_probabilities():
    screen = compose(SnakeGame().snapshot(), {"dead_end_risk": None, "food_reachable": None}, {}).rich_text().plain
    assert screen.count("NOT COMPUTED") == 2
    assert "FAST" in screen


def test_full_metrics_retains_old_meaning():
    policy = LayaPolicy.__new__(LayaPolicy)
    policy.agent = FixedInference()
    policy.guarded, policy.prompt, policy.full_metrics = True, "compact", True
    decision = policy.decide(SnakeGame())
    assert decision.dead_end_risk == pytest.approx(.1)
    assert decision.food_reachable == .8
    assert set(policy.agent.questions) == {"move", "risk", "food"}
