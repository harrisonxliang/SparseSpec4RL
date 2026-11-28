"""RL rollout wrappers around the SparseSpec engine.

Nothing heavy is imported here on purpose: ``serve.rl.engine`` pulls in
flashinfer/triton and needs CUDA, while ``serve.rl.reference_engine`` and the
verl glue only need torch/transformers. Import the module you need directly,
or use the lazy attributes below.
"""

from serve.rl.base import GenerationResult, RolloutEngine

__all__ = [
    "GenerationResult",
    "RolloutEngine",
    "ReferenceEngine",
    "SparseSpecEngine",
]


def __getattr__(name):
    if name == "ReferenceEngine":
        from serve.rl.reference_engine import ReferenceEngine

        return ReferenceEngine
    if name == "SparseSpecEngine":
        from serve.rl.engine import SparseSpecEngine

        return SparseSpecEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
