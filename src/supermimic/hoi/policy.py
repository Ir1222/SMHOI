"""Approved fixed-width HOI Actor B and independent Critic B+.

Original Policy/Critic classes and defaults are unchanged. Only this opt-in
schema exposes Object feedback; no contact/force tensors enter either network.
"""

import torch
from torch import nn

from supermimic.hoi.policy_features import (
    SCHEMA,
    assemble_critic_input,
    feature_schema,
    object_scale,
    width,
)
from supermimic.policy.nets import MotionEncoder, PolicyCfg, _goal_scale_vector, _obs_scale_vector
from supermimic.world_model.dynamics import MLP


class HOIActor(nn.Module):
    schema = SCHEMA

    def __init__(self):
        super().__init__()
        robot_cfg = PolicyCfg()
        self.robot_encoder = MotionEncoder(robot_cfg)
        self.object_history_encoder = MLP(150, (128,), 64)
        self.object_goal_encoder = MLP(150, (128, 64), 32)
        self.decoder = MLP(
            1058, robot_cfg.decoder_hidden, 29, last_gain=robot_cfg.decoder_last_gain
        )
        self.register_buffer("robot_history_scale", _obs_scale_vector(robot_cfg))
        self.register_buffer("object_history_scale", object_scale(history=True))
        self.register_buffer("object_goal_scale", object_scale(future=True))

    def decoder_input(self, observation, goal):
        width(observation, 1080)
        width(goal, 830)
        if observation.shape[:-1] != goal.shape[:-1]:
            raise ValueError("Actor observations/goals must align")
        return torch.cat(
            (
                self.robot_encoder(goal[..., :680]),
                observation[..., :930] * self.robot_history_scale,
                self.object_history_encoder(observation[..., 930:] * self.object_history_scale),
                torch.tanh(self.object_goal_encoder(goal[..., 680:] * self.object_goal_scale)),
            ),
            -1,
        )

    def forward(self, observation, goal):
        return self.decoder(self.decoder_input(observation, goal))

    @staticmethod
    def metadata():
        return {
            "features": feature_schema(),
            "robot_encoder": [680, 2048, 1024, 512, 256, 32],
            "object_history_encoder": [150, 128, 64],
            "object_goal_encoder": [150, 128, 64, 32],
            "decoder": [1058, 2048, 1024, 512, 256, 29],
            "activation": "SiLU hidden; tanh goal latents; linear history embedding/action",
            "decoder_last_gain": 0.01,
        }


class HOICritic(nn.Module):
    schema = SCHEMA

    def __init__(self):
        super().__init__()
        self.mlp = MLP(1015, (256, 128), 1)
        self.register_buffer("robot_goal_scale", _goal_scale_vector(PolicyCfg(), 680))
        self.register_buffer("object_goal_scale", object_scale(future=True))
        self.register_buffer("object_error_scale", object_scale())

    def input_tensor(self, features, goals, xi_norm, root_error):
        return assemble_critic_input(
            features,
            goals,
            xi_norm,
            root_error,
            error_scale=self.object_error_scale,
            robot_goal_scale=self.robot_goal_scale,
            object_goal_scale=self.object_goal_scale,
        )

    def forward(self, features, goals, xi_norm, root_error):
        return self.mlp(self.input_tensor(features, goals, xi_norm, root_error)).squeeze(-1)

    @staticmethod
    def metadata():
        return {
            "features": feature_schema(),
            "mlp": [1015, 256, 128, 1],
            "activation": "SiLU hidden; linear scalar value",
            "actor_weight_sharing": False,
        }
