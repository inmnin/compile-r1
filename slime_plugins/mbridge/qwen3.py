from mbridge.core import register_model
from mbridge.models import Qwen3Bridge as _BaseQwen3Bridge


@register_model("qwen3")
class Qwen3Bridge(_BaseQwen3Bridge):
    """Patch qwen3 mapping for local-spec Megatron parameter names."""

    _MLP_MAPPING = _BaseQwen3Bridge._MLP_MAPPING | {
        # local-spec transformer layer norm naming
        "pre_mlp_layernorm.weight": ["model.layers.{layer_number}.post_attention_layernorm.weight"],
    }

    _OTHER_MAPPING = getattr(_BaseQwen3Bridge, "_OTHER_MAPPING", {}) | {
        # local-spec transformer layer norm naming
        "input_layernorm.weight": ["model.layers.{layer_number}.input_layernorm.weight"],
    }
