from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

try:
    from scipy.spatial import cKDTree
except ImportError:  # scipy is optional; exact search remains the dependency-free path.
    cKDTree = None

from MotionMatching.MotionMatchingConfig import (
    MM_ACTION_FILTER_HARD,
    MM_ACTION_FILTER_MODES,
    MM_ACTION_FILTER_OFF,
    MM_ACTION_FILTER_SOFT,
    MM_ACTION_HARD_THRESHOLD,
    MM_ACTION_MIN_CANDIDATES,
    MM_ACTION_SOFT_PENALTY,
    MM_CURRENT_FRAME_BIAS,
    MM_DEFAULT_ACTION_FILTER_MODE,
    MM_DEFAULT_SEARCH_BACKEND,
    MM_IGNORE_RANGE_END_FRAMES,
    MM_IGNORE_SURROUNDING_FRAMES,
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
    ignore_surrounding_frames: int = MM_IGNORE_SURROUNDING_FRAMES
    ignore_range_end_frames: int = MM_IGNORE_RANGE_END_FRAMES
    action_filter_mode: str = MM_DEFAULT_ACTION_FILTER_MODE
    action_hard_threshold: float = MM_ACTION_HARD_THRESHOLD
    action_min_candidates: int = MM_ACTION_MIN_CANDIDATES
    action_soft_penalty: float = MM_ACTION_SOFT_PENALTY
    kd_min_samples: int = MM_KDTREE_MIN_SAMPLES
    kd_leaf_size: int = MM_KDTREE_LEAF_SIZE
    kd_query_oversample: int = MM_KDTREE_QUERY_OVERSAMPLE
    kd_eps: float = MM_KDTREE_EPS

    def __post_init__(self) -> None:
        if self.backend not in MM_SEARCH_BACKENDS:
            raise ValueError(f"Unsupported search backend: {self.backend!r}. Choices: {MM_SEARCH_BACKENDS}.")
        if self.action_filter_mode not in MM_ACTION_FILTER_MODES:
            raise ValueError(
                f"Unsupported action_filter_mode: {self.action_filter_mode!r}. "
                f"Choices: {MM_ACTION_FILTER_MODES}."
            )
        if int(self.ignore_surrounding_frames) < 0:
            raise ValueError("ignore_surrounding_frames must be non-negative.")
        if int(self.ignore_range_end_frames) < 0:
            raise ValueError("ignore_range_end_frames must be non-negative.")
        if float(self.action_hard_threshold) < 0.0 or float(self.action_hard_threshold) > 1.0:
            raise ValueError("action_hard_threshold must be in [0, 1].")
        if int(self.action_min_candidates) < 0:
            raise ValueError("action_min_candidates must be non-negative.")
        if float(self.action_soft_penalty) < 0.0:
            raise ValueError("action_soft_penalty must be non-negative.")
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
    filtered_candidate_count: int = 0
    action_affinity: float = 0.0
    action_filter_mode: str = MM_ACTION_FILTER_OFF


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
        action_weights: np.ndarray | None,
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
        action_weights_input = None if action_weights is None else np.asarray(action_weights, dtype=np.float32)
        if action_weights_input is not None and action_weights_input.shape[0] != self.features.shape[0]:
            raise ValueError(
                "action_weights sample count does not match features: "
                f"{action_weights_input.shape[0]} vs {self.features.shape[0]}."
            )

        self.range_starts = np.asarray(range_starts, dtype=np.int32)
        self.range_stops = np.asarray(range_stops, dtype=np.int32)
        if len(self.range_starts) != len(self.range_stops):
            raise ValueError("range_starts and range_stops length mismatch.")

        self.action_labels = tuple(str(label) for label in action_labels)
        if action_weights_input is None:
            self.action_weights = np.zeros((len(self.action_ids), len(self.action_labels)), dtype=np.float32)
            valid_ids = (self.action_ids >= 0) & (self.action_ids < len(self.action_labels))
            self.action_weights[np.arange(len(self.action_ids))[valid_ids], self.action_ids[valid_ids]] = 1.0
        else:
            self.action_weights = action_weights_input
        self.range_names = (
            tuple(f"range_{index}" for index in range(len(self.range_starts)))
            if range_names is None
            else tuple(str(name) for name in range_names)
        )
        if len(self.range_names) != len(self.range_starts):
            raise ValueError("range_names length does not match range bounds.")
        if self.action_weights.shape[1] != len(self.action_labels):
            raise ValueError(
                "action_weights action dimension does not match action_labels: "
                f"{self.action_weights.shape[1]} vs {len(self.action_labels)}."
            )

        self.config = config or SearchConfig()
        self.all_indices = np.arange(len(self.features), dtype=np.int32)
        self.range_ids = self._build_range_ids()
        self.frames_until_range_end = self._build_frames_until_range_end()
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
            dataset.action_weights,
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

    def _build_frames_until_range_end(self) -> np.ndarray:
        frames = np.zeros(self.sample_count, dtype=np.int32)
        for start, stop in zip(self.range_starts, self.range_stops):
            indices = np.arange(int(start), int(stop), dtype=np.int32)
            frames[int(start):int(stop)] = int(stop) - indices - 1
        return frames

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

    def get_frames_until_range_end(self, sample_index: int) -> int:
        sample_index = int(sample_index)
        if sample_index < 0 or sample_index >= self.sample_count:
            raise IndexError(f"Sample index out of range: {sample_index}")
        return int(self.frames_until_range_end[sample_index])

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
        query_count = min(
            len(bucket.indices),
            max(1, 1 + int(self.config.kd_query_oversample), min(len(candidates), 64)),
        )
        _, local_indices = bucket.tree.query(query_feature, k=query_count, eps=float(self.config.kd_eps))
        local_indices = np.asarray(local_indices, dtype=np.int64).reshape(-1)
        valid = (local_indices >= 0) & (local_indices < len(bucket.indices))
        selected = bucket.indices[local_indices[valid]].astype(np.int32)
        selected = selected[np.isin(selected, candidates)].astype(np.int32)

        if apply_current_bias and current_index is not None and float(self.config.current_frame_bias) > 0.0:
            current_index = int(current_index)
            if np.any(candidates == current_index):
                selected = np.concatenate([selected, np.asarray([current_index], dtype=np.int32)])

        return self._unique_preserve_order(selected)

    def _base_candidates(self, current_index: int | None) -> np.ndarray:
        mask = np.ones(self.sample_count, dtype=bool)

        ignore_range_end_frames = int(self.config.ignore_range_end_frames)
        if ignore_range_end_frames > 0:
            mask &= self.frames_until_range_end >= ignore_range_end_frames

        ignore_surrounding_frames = int(self.config.ignore_surrounding_frames)
        if current_index is not None and ignore_surrounding_frames > 0:
            current_index = int(current_index)
            mask &= np.abs(self.all_indices - current_index) > ignore_surrounding_frames

        candidates = self.all_indices[mask]
        return candidates if len(candidates) > 0 else self.all_indices

    def _filter_candidate_indices(
        self,
        candidate_indices: np.ndarray,
        current_index: int | None,
    ) -> np.ndarray:
        candidate_indices = self._unique_preserve_order(
            np.asarray(candidate_indices, dtype=np.int32).reshape(-1)
        )
        if len(candidate_indices) == 0:
            return candidate_indices
        if np.any(candidate_indices < 0) or np.any(candidate_indices >= self.sample_count):
            raise ValueError("candidate_indices contains samples outside the search index.")

        mask = np.ones(len(candidate_indices), dtype=bool)
        ignore_range_end_frames = int(self.config.ignore_range_end_frames)
        if ignore_range_end_frames > 0:
            mask &= self.frames_until_range_end[candidate_indices] >= ignore_range_end_frames

        ignore_surrounding_frames = int(self.config.ignore_surrounding_frames)
        if current_index is not None and ignore_surrounding_frames > 0:
            mask &= np.abs(candidate_indices - int(current_index)) > ignore_surrounding_frames

        filtered = candidate_indices[mask].astype(np.int32)
        return filtered if len(filtered) > 0 else candidate_indices

    def _normalized_action_weights(self, action_weights: np.ndarray | None) -> np.ndarray | None:
        if action_weights is None or self.config.action_filter_mode == MM_ACTION_FILTER_OFF:
            return None
        weights = np.asarray(action_weights, dtype=np.float32).reshape(-1)
        if weights.shape[0] != len(self.action_labels):
            raise ValueError(
                f"action_weights dim {weights.shape[0]} does not match {len(self.action_labels)}."
            )
        if not np.all(np.isfinite(weights)):
            raise ValueError("action_weights contains non-finite values.")
        weights = np.maximum(weights, 0.0)
        total = float(np.sum(weights))
        if total <= 1e-8:
            return None
        return (weights / total).astype(np.float32)

    def _action_affinities(self, candidates: np.ndarray, action_weights: np.ndarray | None) -> np.ndarray:
        if action_weights is None:
            return np.zeros(len(candidates), dtype=np.float32)
        return np.clip(self.action_weights[candidates] @ action_weights, 0.0, 1.0).astype(np.float32)

    def _apply_action_hard_filter(
        self,
        candidates: np.ndarray,
        action_weights: np.ndarray | None,
    ) -> tuple[np.ndarray, bool]:
        if (
            action_weights is None
            or self.config.action_filter_mode != MM_ACTION_FILTER_HARD
            or len(candidates) == 0
        ):
            return candidates, False

        affinities = self._action_affinities(candidates, action_weights)
        hard_candidates = candidates[affinities >= float(self.config.action_hard_threshold)].astype(np.int32)
        if len(hard_candidates) >= int(self.config.action_min_candidates):
            return hard_candidates, True
        return candidates, False

    def _rank_candidates(
        self,
        query_feature: np.ndarray,
        candidates: np.ndarray,
        current_index: int | None,
        apply_current_bias: bool,
        candidate_count: int,
        backend: str,
        action_weights: np.ndarray | None,
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

        affinities = self._action_affinities(candidates, action_weights)
        if (
            action_weights is not None
            and self.config.action_filter_mode in (MM_ACTION_FILTER_SOFT, MM_ACTION_FILTER_HARD)
            and float(self.config.action_soft_penalty) > 0.0
        ):
            scores += float(self.config.action_soft_penalty) * (1.0 - affinities)

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
            filtered_candidate_count=int(len(candidates)),
            action_affinity=float(affinities[best_local]) if len(affinities) else 0.0,
            action_filter_mode=str(self.config.action_filter_mode),
        )

    def score_candidate(
        self,
        query_feature: np.ndarray,
        sample_index: int,
        action_weights: np.ndarray | None = None,
    ) -> float:
        query_feature = np.asarray(query_feature, dtype=np.float32).reshape(-1)
        sample_index = int(sample_index)
        delta = self.features[sample_index] - query_feature
        score = float(np.sqrt(max(float(np.dot(delta, delta)), 0.0)))
        normalized_action_weights = self._normalized_action_weights(action_weights)
        if (
            normalized_action_weights is not None
            and self.config.action_filter_mode in (MM_ACTION_FILTER_SOFT, MM_ACTION_FILTER_HARD)
            and float(self.config.action_soft_penalty) > 0.0
        ):
            affinity = float(self._action_affinities(np.asarray([sample_index], dtype=np.int32), normalized_action_weights)[0])
            score += float(self.config.action_soft_penalty) * (1.0 - affinity)
        return score

    def search(
        self,
        query_feature: np.ndarray,
        current_index: int | None = None,
        apply_current_bias: bool = True,
        action_weights: np.ndarray | None = None,
        candidate_indices: np.ndarray | None = None,
    ) -> SearchResult:
        query_feature = np.asarray(query_feature, dtype=np.float32).reshape(-1)
        if query_feature.shape[0] != self.feature_dim:
            raise ValueError(f"query_feature dim {query_feature.shape[0]} does not match {self.feature_dim}.")

        action_weights = self._normalized_action_weights(action_weights)
        if candidate_indices is None:
            candidates = self._base_candidates(current_index)
        else:
            candidates = self._filter_candidate_indices(candidate_indices, current_index)
        candidates, hard_filtered = self._apply_action_hard_filter(candidates, action_weights)
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
                    action_weights,
                )

        return self._rank_candidates(
            query_feature,
            candidates,
            current_index,
            apply_current_bias,
            len(candidates),
            MM_SEARCH_BACKEND_EXACT if not hard_filtered else f"{MM_SEARCH_BACKEND_EXACT}:hard",
            action_weights,
        )

    def should_transition(self, current_score: float, best_result: SearchResult) -> bool:
        return float(best_result.score) + float(self.config.min_improvement) < float(current_score)
