from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class Feature:
    feature_id: str
    dependencies: tuple[str, ...] = ()
    ensure_clip: Callable | None = None
    dispose_clip: Callable | None = None
    is_ready: Callable | None = None


@dataclass
class FeatureRegistry:
    features: dict[str, Feature]
    ensured_clip_features: list[str] = field(default_factory=list)
    mounted_clip_features: list[str] = field(default_factory=list)

    def ensure_clip(self, app, feature_id):
        feature = self.features[feature_id]
        for dependency_id in feature.dependencies:
            self.ensure_clip(app, dependency_id)

        result = None
        if feature.ensure_clip is not None:
            result = feature.ensure_clip(app)
        if feature_id not in self.ensured_clip_features:
            self.ensured_clip_features.append(feature_id)
        return result

    def ensure_many_clip(self, app, feature_ids):
        return [self.ensure_clip(app, feature_id) for feature_id in feature_ids]

    def is_clip_ready(self, app, feature_id):
        feature = self.features[feature_id]
        if feature.is_ready is not None:
            return bool(feature.is_ready(app))
        return feature_id in self.ensured_clip_features

    def mount_clip(self, app, feature_id):
        result = self.ensure_clip(app, feature_id)
        if feature_id not in self.mounted_clip_features:
            self.mounted_clip_features.append(feature_id)
        return result

    def unmount_clip(self, app, feature_id):
        if feature_id not in self.mounted_clip_features:
            return
        feature = self.features[feature_id]
        if feature.dispose_clip is not None:
            feature.dispose_clip(app)
        self.mounted_clip_features.remove(feature_id)

    def sync_clip_mount(self, app, feature_id, should_mount):
        if should_mount:
            return self.mount_clip(app, feature_id)
        self.unmount_clip(app, feature_id)
        return None

    def dispose_clip(self, app):
        disposed_feature_ids = set()
        for feature_id in reversed(list(self.mounted_clip_features)):
            feature = self.features[feature_id]
            if feature.dispose_clip is not None:
                feature.dispose_clip(app)
                disposed_feature_ids.add(feature_id)
        self.mounted_clip_features.clear()
        for feature_id in reversed(list(self.ensured_clip_features)):
            if feature_id in disposed_feature_ids:
                continue
            feature = self.features[feature_id]
            if feature.dispose_clip is not None:
                feature.dispose_clip(app)
        self.ensured_clip_features.clear()
