"""Regression test for the optimizer.py syntax error (two dangling `try`
blocks with no `except`/`finally`) that used to make the whole module
unimportable."""


def test_optimizer_module_imports():
    from src.utils import optimizer  # noqa: F401


def test_walk_forward_optimizer_importable():
    from src.utils.optimizer import StrategyOptimizer, WalkForwardOptimizer  # noqa: F401
