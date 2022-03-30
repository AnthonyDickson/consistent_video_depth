#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

from .depth_model import DepthModel
from .mannequin_challenge_model import MannequinChallengeModel

from typing import List


def get_depth_model_list() -> List[str]:
    return ["mc"]


def get_depth_model(type: str) -> DepthModel:
    if type == "mc":
        return MannequinChallengeModel
    else:
        raise ValueError(f"Unsupported model type '{type}'.")


def create_depth_model(type: str) -> DepthModel:
    model_class = get_depth_model(type)
    return model_class()
