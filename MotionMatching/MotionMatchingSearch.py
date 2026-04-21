from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

try:
    from scipy.spatial import cKDTree
except ImportError:  # scipy is optional; exact search remains the dependency-free path.
    cKDTree = None

from MotionMatching.MotionMatchingConfig import (
    MM_CURRENT_FRAME_BIAS,
    MM_DEFAULT_SEARCH_BACKEND,
    MM_KDTREE_EPS,
    MM_KDTREE_LEAF_SIZE,
    MM_KDTREE_MIN_SAMPLES,
    MM_KDTREE_QUERY_OVERSAMPLE,
    MM_MIN_IMPROVEMENT,
    MM_SEARCH_BACKEND_AUTO,
    MM_SEARCH_BACKEND_EXACT,
    MM_SEARCH_BACKEND_KDTREE,
    MM_SEARCH_BACKENDS,
)
from MotionMatching.MotionMatchingDataset import MotionMatchingDataset


@dataclass(frozen=True)
class SearchConfig:
    backend: str = MM_DEFAULT_SEARCH_BACKEND
    current_frame_bias: float = MM_CURRENT_FRAME_BIAS
    min_improvement: float = MM_MIN_IMPROVEMENT
    kd_min_samples: int = MM_KDTREE_MIN_SAMPLES
    kd_leaf_size: int = MM_KDTREE_LEAF_SIZE
    kd_query_oversample: int = MM_KDTREE_QUERY_OVERSAMPLE
    kd_eps: float = MM_KDTREE_EPS

    def __post_init__(self) -> None:
        if self.backend not in MM_SEARCH_BACKENDS:
            raise ValueError(f"Unsupported search backend: {self.backend!r}. Choices: {MM_SEARCH_BACKENDS}.")
        if int(self.kd_min_samples) < 0:
            raise ValueError("kd_min_samples must be non-negative.")
        if int(self.kd_leaf_size) <= 0:
            raise ValueError("kd_leaf_size must be positive.")
        if int(self.kd_query_oversample) < 0:
            raise ValueError("kd_query_oversample must be non-negative.")
        if float(self.kd_eps) < 0.0:
            raise ValueError("kd_eps must be non-negative.")


@dataclass(frozen=True)
class SearchResult:
    index: int
    distance: float
    score: float
    action_id: int
    action_label: str
    range_index: int
    range_name: str
    candidate_count: int
    backend: str = MM_SEARCH_BACKEND_EXACT


@dataclass(frozen=True)
class _KDTreeBucket:
    indices: np.ndarray
    tree: object


class MotionMatchingSearchIndex:
    """Nearest-neighbor search over weighted Motion Matching features."""

    def __init__(
        self,
        features: np.ndarray,
        action_ids: np.ndarray,
        range_starts: np.ndarray,
        range_stops: np.ndarray,
        action_labels: Iterable[str],
        range_names: Iterable[str] | None = None,
        config: SearchConfig | None = None,
    ) -> None:
        self.features = np.asarray(features, dtype=np.float32)
        if self.features.ndim != 2:
            raise ValueError(f"features must be a 2D matrix, got {self.features.shape}.")

        self.action_ids = np.asarray(action_ids, dtype=np.int32)
        if self.action_ids.shape[0] != self.features.shape[0]:
            raise ValueError(
                "action_ids sample count does not match features: "
                f"{self.action_ids.shape[0]} vs {self.features.shape[0]}."
            )

        self.range_starts = np.asarray(range_starts, dtype=np.int32)
        self.range_stops = np.asarray(range_stops, dtype=np.int32)
        if len(self.range_starts) != len(self.range_stops):
            raise ValueError("range_starts and range_stops length mismatch.")

        self.action_labels = tuple(str(label) for label in action_labels)
        self.range_names = (
            tuple(f"range_{index}" for index in range(len(self.range_starts)))
            if range_names is None
            else tuple(str(name) for name in range_names)
        )
        if len(self.range_names) != len(self.range_starts):
            raise ValueError("range_names length does not match range bounds.")

        self.config = config or SearchConfig()
        self.all_indices = np.arange(len(self.features), dtype=np.int32)
        self.range_ids = self._build_range_ids()
        self.action_buckets = self._build_action_buckets()
        self._kdtree_action_buckets: dict[int, _KDTreeBucket] = {}
        self._kdtree_all_bucket: _KDTreeBucket | None = None
        if self.config.backend == MM_SEARCH_BACKEND_KDTREE and cKDTree is None:
            raise ImportError("SearchConfig backend='kdtree' requires scipy.spatial.cKDTree.")

    @classmethod
    def from_dataset(
        cls,
        dataset: MotionMatchingDataset,
        config: SearchConfig | None = None,
    ) -> "MotionMatchingSearchIndex":
        return cls(
            dataset.features,
            dataset.action_ids,
            dataset.range_starts,
            dataset.range_stops,
            dataset.action_labels,
            range_names=dataset.range_names,
            config=config,
        )

    @property
    def sample_count(self) -> int:
        return int(self.features.shape[0])

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[1])

    def _build_range_ids(self) -> np.ndarray:
        range_ids = np.full(self.sample_count, -1, dtype=np.int32)
        for range_index, (start, stop) in enumerate(zip(self.range_starts, self.range_stops)):
            if start < 0 or stop > self.sample_count or start >= stop:
                raise ValueError(f"Invalid range bounds: [{start}:{stop}]")
            range_ids[int(start):int(stop)] = int(range_index)
        if np.any(range_ids < 0):
            raise ValueError("Some feature rows are not covered by any range.")
        return range_ids

    def _build_action_buckets(self) -> dict[int, np.ndarray]:
        return {
            action_id: np.flatnonzero(self.action_ids == action_id).astype(np.int32)
            for action_id in range(len(self.action_labels))
        }

    def resolve_action_id(self, action: int | str | None) -> int | None:
        if action is None:
            return None
        if isinstance(action, str):
            if action not in self.action_labels:
                raise ValueError(f"Unknown action label: {action}")
            return int(self.action_labels.index(action))
        action_id = int(action)
        if action_id < 0 or action_id >= len(self.action_labels):
            raise ValueError(f"Invalid action id: {action_id}")
        return action_id

    def get_range_index(self, sample_index: int) -> int:
        sample_index = int(sample_index)
        if sample_index < 0 or sample_index >= self.sample_count:
            raise IndexError(f"Sample index out of range: {sample_index}")
        return int(self.range_ids[sample_index])

    def get_next_index(self, sample_index: int, step: int = 1) -> int | None:
        sample_index = int(sample_index)
        range_index = self.get_range_index(sample_index)
        next_index = sample_index + int(step)
        if next_index >= int(self.range_stops[range_index]):
            return None
        return int(next_index)

    @staticmethod
    def _unique_preserve_order(indices: np.ndarray) -> np.ndarray:
        seen = set()
        unique = []
        for index in np.asarray(indices, dtype=np.int32).reshape(-1):
            value = int(index)
            if value in seen:
                continue
            seen.add(value)
            unique.append(value)
        return np.asarray(unique, dtype=np.int32)

    def _build_kdtree_bucket(self, indices: np.ndarray) -> _KDTreeBucket:
        if cKDTree is None:
            raise ImportError("scipy.spatial.cKDTree is not available.")
        indices = np.asarray(indices, dtype=np.int32)
        return _KDTreeBucket(
            indices=indices,
            tree=cKDTree(self.features[indices], leafsize=int(self.config.kd_leaf_size)),
        )

    def _get_kdtree_bucket(self, action_id: int | None) -> _KDTreeBucket:
        if action_id is None:
            if self._kdtree_all_bucket is None:
                self._kdtree_all_bucket = self._build_kdtree_bucket(self.all_indices)
            return self._kdtree_all_bucket

        action_id = int(action_id)
        if action_id not in self._kdtree_action_buckets:
            self._kdtree_action_buckets[action_id] = self._build_kdtree_bucket(self.action_buckets[action_id])
        return self._kdtree_action_buckets[action_id]

    def _should_use_kdtree(self, candidates: np.ndarray) -> bool:
        if self.config.backend == MM_SEARCH_BACKEND_EXACT:
            return False
        if cKDTree is None:
            if self.config.backend == MM_SEARCH_BACKEND_KDTREE:
                raise ImportError("SearchConfig backend='kdtree' requires scipy.spatial.cKDTree.")
            return False
        if self.config.backend == MM_SEARCH_BACKEND_KDTREE:
            return True
        if self.config.backend == MM_SEARCH_BACKEND_AUTO:
            return len(candidates) >= int(self.config.kd_min_samples)
        return False

    def _query_kdtree_candidates(
        self,
        query_feature: np.ndarray,
        candidates: np.ndarray,
        action_id: int | None,
        current_index: int | None,
        apply_current_bias: bool,
    ) -> np.ndarray:
        bucket = self._get_kdtree_bucket(action_id)
        query_count = min(len(candidates), max(1, 1 + int(self.config.kd_query_oversample)))
        _, local_indices = bucket.tree.query(query_feature, k=query_count, eps=float(self.config.kd_eps))
        local_indices = np.asarray(local_indices, dtype=np.int64).reshape(-1)
        valid = (local_indices >= 0) & (local_indices < len(bucket.indices))
        selected = bucket.indices[local_indices[valid]].astype(np.int32)

        if apply_current_bias and current_index is not None and float(self.config.current_frame_bias) > 0.0:
            current_index = int(current_index)
            if np.any(candidates == current_index):
                selected = np.concatenate([selected, np.asarray([current_index], dtype=np.int32)])

        return self._unique_preserve_order(selected)

    def _rank_candidates(
        self,
        query_feature: np.ndarray,
        candidates: np.ndarray,
        current_index: int | None,
        apply_current_bias: bool,
        candidate_count: int,
        backend: str,
    ) -> SearchResult:
        if len(candidates) == 0:
            raise ValueError("No candidates available for Motion Matching search.")

        deltas = self.features[candidates] - query_feature[np.newaxis, :]
        distances_squared = np.einsum("ij,ij->i", deltas, deltas, dtype=np.float32)
        distances = np.sqrt(np.maximum(distances_squared, 0.0)).astype(np.float32)
        scores = distances.copy()

        if apply_current_bias and current_index is not None and float(self.config.current_frame_bias) > 0.0:
            current_mask = candidates == int(current_index)
            if np.any(current_mask):
                scores[current_mask] -= float(self.config.current_frame_bias)

        best_local = int(np.argmin(scores))
        index = int(candidates[best_local])
        action_id = int(self.action_ids[index])
        range_index = int(self.range_ids[index])
        return SearchResult(
            index=index,
            distance=float(distances[best_local]),
            score=float(scores[best_local]),
            action_id=action_id,
            action_label=self.action_labels[action_id],
            range_index=range_index,
            range_name=self.range_names[range_index],
            candidate_count=int(candidate_count),
            backend=str(backend),
        )

    def search(
        self,
        query_feature: np.ndarray,
        current_index: int | None = None,
        apply_current_bias: bool = True,
    ) -> SearchResult:
        query_feature = np.asarray(query_feature, dtype=np.float32).reshape(-1)
        if query_feature.shape[0] != self.feature_dim:
            raise ValueError(f"query_feature dim {query_feature.shape[0]} does not match {self.feature_dim}.")

        candidates = self.all_indices
        action_id = None
        if self._should_use_kdtree(candidates):
            kdtree_candidates = self._query_kdtree_candidates(
                query_feature,
                candidates,
                action_id,
                current_index,
                apply_current_bias,
            )
            if len(kdtree_candidates) > 0:
                return self._rank_candidates(
                    query_feature,
                    kdtree_candidates,
                    current_index,
                    apply_current_bias,
                    len(candidates),
                    MM_SEARCH_BACKEND_KDTREE,
                )

        return self._rank_candidates(
            query_feature,
            candidates,
            current_index,
            apply_current_bias,
            len(candidates),
            MM_SEARCH_BACKEND_EXACT,
        )

    def should_transition(self, current_distance: float, best_result: SearchResult) -> bool:
        return float(best_result.score) + float(self.config.min_improvement) < float(current_distance)
