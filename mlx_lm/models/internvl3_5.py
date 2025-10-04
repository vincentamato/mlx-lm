# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from . import gpt_oss, qwen3, qwen3_moe
from .base import BaseModelArgs
from .cache import KVCache


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict

    @classmethod
    def from_dict(cls, params):
        return cls(
            model_type=params["model_type"],
            text_config=params.get("text_config", {}).copy(),
        )


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type

        text_model_map = {
            "qwen3": qwen3,
            "qwen3_moe": qwen3_moe,
            "gpt_oss": gpt_oss,
        }

        text_model_type = args.text_config.get("model_type")
        if text_model_type not in text_model_map:
            raise ValueError(f"Unsupported text model type: {text_model_type}")

        text_model_module = text_model_map[text_model_type]

        # Add tie_word_embeddings if missing
        if "tie_word_embeddings" not in args.text_config:
            args.text_config["tie_word_embeddings"] = False

        self.language_model = text_model_module.Model(
            text_model_module.ModelArgs.from_dict(args.text_config)
        )

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        return self.language_model(
            inputs, cache=cache, input_embeddings=input_embeddings
        )

    def sanitize(self, weights):
        # Remove vision components
        weights = {
            k: v
            for k, v in weights.items()
            if not k.startswith(("vision_tower.", "multi_modal_projector."))
        }

        # Handle model-specific weight transformations
        model_type = self.args.text_config.get("model_type")
        if model_type == "gpt_oss":
            # Handle gpt-oss interleaved weights
            normalized = {}
            for k, v in weights.items():
                if "language_model." in k and "mlp.experts.gate_up_proj" in k:
                    base_key = k.replace("gate_up_proj", "")
                    if "bias" in k:
                        # Interleaved bias: even=gate, odd=up
                        normalized[base_key.replace("_bias", "gate_proj.bias")] = v[
                            ..., ::2
                        ]
                        normalized[base_key.replace("_bias", "up_proj.bias")] = v[
                            ..., 1::2
                        ]
                    else:
                        # Interleaved weights: even=gate, odd=up (after transpose)
                        v_t = v.transpose(0, 2, 1)
                        normalized[base_key + "gate_proj.weight"] = v_t[:, ::2, :]
                        normalized[base_key + "up_proj.weight"] = v_t[:, 1::2, :]
                # Handle down_proj bias naming convention
                elif "language_model." in k and k.endswith("down_proj_bias"):
                    normalized[k.replace("down_proj_bias", "down_proj.bias")] = v
                # Handle down_proj weights which need transpose and .weight suffix
                elif (
                    "language_model." in k
                    and ".mlp.experts.down_proj" in k
                    and not k.endswith((".weight", ".bias", ".biases", ".scales"))
                ):
                    normalized[k + ".weight"] = v.transpose(0, 2, 1)
                # Keep all other weights as-is
                else:
                    normalized[k] = v
            weights = normalized
        elif model_type == "qwen3_moe":
            # Stack individual expert weights for Qwen3 MoE
            num_layers = self.args.text_config.get("num_hidden_layers", 0)
            num_experts = self.args.text_config.get("num_experts", 0)

            for l in range(num_layers):
                prefix = f"language_model.model.layers.{l}"
                for proj in ["up_proj", "down_proj", "gate_proj"]:
                    first_key = f"{prefix}.mlp.experts.0.{proj}.weight"
                    if first_key in weights:
                        # Stack all expert weights for this projection
                        expert_weights = []
                        for e in range(num_experts):
                            key = f"{prefix}.mlp.experts.{e}.{proj}.weight"
                            if key in weights:
                                expert_weights.append(weights.pop(key))
                        if expert_weights:
                            weights[f"{prefix}.mlp.switch_mlp.{proj}.weight"] = (
                                mx.stack(expert_weights)
                            )

        return weights

    @property
    def layers(self):
        return self.language_model.model.layers

    def make_cache(self):
        if hasattr(self.language_model, "make_cache"):
            return self.language_model.make_cache()
        return [KVCache() for _ in range(len(self.layers))]
