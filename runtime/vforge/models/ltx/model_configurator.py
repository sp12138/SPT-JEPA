import torch

from vforge.models.ltx.embeddings_connector import (
    AudioEmbeddings1DConnectorConfigurator,
    Embeddings1DConnectorConfigurator,
)
from vforge.models.ltx.model import LTXModel, LTXModelType
from vforge.models.ltx.rope import LTXRopeType
from vforge.models.ltx.transformer import DEFAULT_TRANSFORMER_OPS, TransformerOpsConfig


def _check_config_value(config: dict, key: str, expected: object) -> None:
    actual = config.get(key)
    if actual != expected:
        raise ValueError(f"Config value {key} is {actual}, expected {expected}")


class LTXModelConfigurator:
    """
    Configurator for LTX model.
    Used to create an LTX model from a configuration dictionary.
    """

    model_class: type[LTXModel] = LTXModel

    @classmethod
    def from_config(
        cls,
        config: dict,
        ops: TransformerOpsConfig = DEFAULT_TRANSFORMER_OPS,
        *,
        device: torch.device | str | None = None,
    ) -> LTXModel:
        model_config = config
        config = model_config.get("transformer", {})

        for key, expected in (
            ("dropout", 0.0),
            ("attention_bias", True),
            ("num_vector_embeds", None),
            ("activation_fn", "gelu-approximate"),
            ("num_embeds_ada_norm", 1000),
            ("use_linear_projection", False),
            ("only_cross_attention", False),
            ("cross_attention_norm", True),
            ("double_self_attention", False),
            ("upcast_attention", False),
            ("standardization_norm", "rms_norm"),
            ("norm_elementwise_affine", False),
            ("qk_norm", "rms_norm"),
            ("positional_embedding_type", "rope"),
            ("use_audio_video_cross_attention", True),
            ("share_ff", False),
            ("av_cross_ada_norm", True),
            ("use_middle_indices_grid", True),
            ("caption_proj_before_connector", True),
        ):
            _check_config_value(config, key, expected)
        _check_config_value(
            config,
            "num_attention_heads",
            config.get("audio_num_attention_heads", float("nan")),
        )

        kwargs = dict(
            model_type=LTXModelType.AudioVideo,
            num_attention_heads=config.get("num_attention_heads", 32),
            attention_head_dim=config.get("attention_head_dim", 128),
            in_channels=config.get("in_channels", 128),
            out_channels=config.get("out_channels", 128),
            num_layers=config.get("num_layers", 48),
            cross_attention_dim=config.get("cross_attention_dim", 4096),
            norm_eps=config.get("norm_eps", 1e-06),
            ops=ops,
            positional_embedding_theta=config.get("positional_embedding_theta", 10000.0),
            positional_embedding_max_pos=config.get(
                "positional_embedding_max_pos", [20, 2048, 2048]
            ),
            timestep_scale_multiplier=config.get("timestep_scale_multiplier", 1000),
            use_middle_indices_grid=config.get("use_middle_indices_grid", True),
            audio_num_attention_heads=config.get("audio_num_attention_heads", 32),
            audio_attention_head_dim=config.get("audio_attention_head_dim", 64),
            audio_in_channels=config.get("audio_in_channels", 128),
            audio_out_channels=config.get("audio_out_channels", 128),
            audio_cross_attention_dim=config.get("audio_cross_attention_dim", 2048),
            audio_positional_embedding_max_pos=config.get(
                "audio_positional_embedding_max_pos", [20]
            ),
            av_ca_timestep_scale_multiplier=config.get("av_ca_timestep_scale_multiplier", 1),
            rope_type=LTXRopeType(config.get("rope_type", "split")),
            double_precision_rope=config.get("frequencies_precision", False) == "float64",
            apply_gated_attention=config.get("apply_gated_attention", False),
            cross_attention_adaln=config.get("cross_attention_adaln", False),
        )

        def build() -> LTXModel:
            video_connector = Embeddings1DConnectorConfigurator.from_config(
                model_config,
                ops=ops.connector_attention_ops,
            )
            audio_connector = AudioEmbeddings1DConnectorConfigurator.from_config(
                model_config,
                ops=ops.connector_attention_ops,
            )
            return cls.model_class(
                **kwargs,
                video_embeddings_connector=video_connector,
                audio_embeddings_connector=audio_connector,
            )

        if device is None:
            return build()
        with torch.device(device):
            return build()
