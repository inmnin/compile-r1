from abc import ABC, abstractmethod


class HfWeightIteratorBase(ABC):
    @staticmethod
    def create(args, model, **kwargs):
        from .hf_weight_iterator_bridge import HfWeightIteratorBridge
        from .hf_weight_iterator_direct import HfWeightIteratorDirect

        c = {
            "raw": HfWeightIteratorDirect,
            "bridge": HfWeightIteratorBridge,
        }[args.megatron_to_hf_mode]

        # Fallback for newer mbridge APIs that removed legacy bridge export hooks.
        if c is HfWeightIteratorBridge:
            from megatron.bridge import AutoBridge

            bridge = AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)
            has_legacy_export = hasattr(bridge, "get_conversion_tasks") and hasattr(bridge, "export_hf_weights")
            if not has_legacy_export:
                c = HfWeightIteratorDirect

        return c(args, model, **kwargs)

    def __init__(self, args, model, model_name, quantization_config):
        self.args = args
        self.model = model
        self.model_name = model_name
        self.quantization_config = quantization_config

    @abstractmethod
    def get_hf_weight_chunks(self, megatron_local_weights):
        """
        Mental model of the API:
        megatron_model.to_hf_magically().named_parameters()
        """
        raise NotImplementedError
