# Source: https://github.com/ray-project/ray/blob/master/rllib/models/torch/visionnet.py

import numpy as np
from typing import Dict, List
import gym

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import normc_initializer, same_padding, \
    SlimConv2d, SlimFC
from ray.rllib.models.utils import get_filter_config
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType

_, nn = try_import_torch()

class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        print(x.shape)
        return x

class CustomVisionNetwork(TorchModelV2, nn.Module):
    """Generic vision network."""

    def __init__(self, obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)
        # override the layer building earlier
        self._convs = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=10, stride=1, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=5, stride=5),
            nn.Conv2d(3, 3, kernel_size=10, stride=1, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=5, stride=5),
            nn.Flatten(),
            nn.Dropout(),
            # PrintLayer(),
            nn.Linear(12, 256)
        )
        # Holds the current "base" output (before logits layer).
        self._features = None

    @override(TorchModelV2)
    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):
        self._features = input_dict["obs"].float().permute(0, 3, 1, 2)
        conv_out = self._convs(self._features)
        # Store features to save forward pass when getting value_function out.
        self._features = conv_out
        return conv_out, state

    # @override(TorchModelV2)
    # def value_function(self) -> TensorType:
    #     assert self._features is not None, "must call forward() first"
    #     if self._value_branch_separate:
    #         value = self._value_branch_separate(self._features)
    #         value = value.squeeze(3)
    #         value = value.squeeze(2)
    #         return value.squeeze(1)
    #     else:
    #         if not self.last_layer_is_flattened:
    #             features = self._features.squeeze(3)
    #             features = features.squeeze(2)
    #         else:
    #             features = self._features
    #         return self._value_branch(features).squeeze(1)

    # def _hidden_layers(self, obs: TensorType) -> TensorType:
    #     res = self._convs(obs.permute(0, 3, 1, 2))  # switch to channel-major
    #     res = res.squeeze(3)
    #     res = res.squeeze(2)
    #     return res