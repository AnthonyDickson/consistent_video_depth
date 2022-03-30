#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
import os
import warnings
from collections import OrderedDict

import torch
import torch.autograd as autograd

from ..utils.helpers import SuppressedStdout
from ..utils.url_helpers import get_model_from_url

from .mannequin_challenge.models import pix2pix_model
from .mannequin_challenge.options.train_options import TrainOptions
from .depth_model import DepthModel


class MannequinChallengeModel(DepthModel):
    # Requirements and default settings
    align = 16
    learning_rate = 0.0004
    lambda_view_baseline = 0.1

    def __init__(self, model_path_override=None):
        super().__init__()

        parser = TrainOptions()
        parser.initialize()
        params = parser.parser.parse_args(["--input", "single_view"])
        params.isTrain = False

        model_file = model_path_override or get_model_from_url(
            "https://storage.googleapis.com/mannequinchallenge-data/checkpoints/best_depth_Ours_Bilinear_inc_3_net_G.pth",
            local_path="mc.pth",
            path_root=os.environ["WEIGHTS_PATH"]
        )

        class FixedMcModel(pix2pix_model.Pix2PixModel):
            # Override the load function, so we can load the snapshot stored
            # in our specific location.
            def load_network(self, network, network_label, epoch_label):
                return torch.load(model_file)

        with SuppressedStdout():
            try:
                self.model = FixedMcModel(params)
            except RuntimeError:
                warnings.warn(f"Could not create model. Attempting state_dict fix...")
                # Sometimes a model will fail to load because the state_dict keys have the 'module.' prefix.
                # This is because the weights were from a model that was wrapped in torch.nn.DataParallel(...),
                # but the new model that is loading the weights has not been wrapped by this and therefore does not
                # expect the prefix.
                # We can try fix this error by rebuilding the state dict, removing the 'module.' prefix from the keys.
                original_state_dict = torch.load(model_file)

                # Temporarily replace weights file with fixed version.
                fixed_state_dict = OrderedDict(
                    [(key.replace("module.", ""), value) for key, value in original_state_dict.items()]
                )
                torch.save(fixed_state_dict, model_file)

                try:
                    self.model = FixedMcModel(params)
                finally:
                    # Revert changes made to state_dict.
                    torch.save(original_state_dict, model_file)

    def train(self):
        self.model.switch_to_train()

    def eval(self):
        self.model.switch_to_eval()

    def parameters(self):
        return self.model.netG.parameters()

    def estimate_depth(self, images):
        images = autograd.Variable(images.cuda(), requires_grad=False)

        # Reshape ...CHW -> XCHW
        shape = images.shape
        C, H, W = shape[-3:]
        images = images.reshape(-1, C, H, W)

        self.model.prediction_d, _ = self.model.netG.forward(images)

        # Reshape X1HW -> BNHW
        out_shape = shape[:-3] + self.model.prediction_d.shape[-2:]
        self.model.prediction_d = self.model.prediction_d.reshape(out_shape)

        self.model.prediction_d = torch.exp(self.model.prediction_d)
        self.model.prediction_d = self.model.prediction_d.squeeze(-3)

        return self.model.prediction_d

    def save(self, file_name):
        state_dict = self.model.netG.state_dict()
        torch.save(state_dict, file_name)
