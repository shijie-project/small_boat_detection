"""Feature registry -- one module per tab.

To add a feature: drop a module here with a :class:`~webui.features.base.Feature`
subclass (fields + ``build``), then list it in ``FEATURES`` below.
"""

from .base import Feature, Field, JobSpec
from .label_studio import LabelStudioFeature
from .ls_coco import LabelStudioCocoFeature
from .ls_import import LabelStudioImportFeature
from .ls_review import LabelStudioReviewFeature
from .manual_split import ManualSplitFeature
from .split_coco import SplitFeature
from .test import TestFeature
from .train import TrainFeature


FEATURES = [
    TrainFeature(),
    TestFeature(),
    # InferenceFeature() (tiled inference over split_images/) is off for now;
    # its module stays, LS review reuses its form helpers.
    ManualSplitFeature(),
    LabelStudioFeature(),
    LabelStudioImportFeature(),
    LabelStudioReviewFeature(),
    LabelStudioCocoFeature(),
    SplitFeature(),
]


def all_features():
    return list(FEATURES)


def get(name):
    for feature in FEATURES:
        if feature.name == name:
            return feature
    return None
