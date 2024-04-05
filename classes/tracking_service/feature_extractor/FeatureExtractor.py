import torch,cv2,time,os,numpy as np
from typing import Dict, Iterable, Callable
from torch import nn, Tensor
class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, target_layer_names: list):
        super().__init__()
        self.model = model
        self.target_layer_names = target_layer_names
        self.target_layers = {}
        self.layer_outputs = {}
        self._register_hooks(self.model)

    def _register_hooks(self, module: nn.Module, prefix=""):
        # Recursively register hooks for all child modules
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            if full_name in self.target_layer_names:
                self.target_layers[full_name] = child

                # Register a hook to store the output of the target layer
                def hook_fn(name, module, input, output):
                    self.layer_outputs[name] = output
                hook = child.register_forward_hook(lambda m, i, o, name=full_name: hook_fn(name, m, i, o))

        # Recursively register hooks for all child modules
        for name, child in module.named_children():
            self._register_hooks(child, full_name)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def get_layer_output(self, layer_name: str) -> Tensor:
        return self.layer_outputs.get(layer_name, None)

    def _remove_hooks(self):
        for name, layer in self.target_layers.items():
            layer._forward_hooks.clear()

    def __del__(self):
        self._remove_hooks()