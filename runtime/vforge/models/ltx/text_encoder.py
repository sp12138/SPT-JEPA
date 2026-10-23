"""Local-only Gemma 3 prompt encoding; no VAE or training dependencies."""

import json
from functools import partial

import torch
from safetensors import safe_open

import vforge.ops as vforge_ops
from vforge.models.ltx.attention import AttentionOps
from vforge.models.ltx.embeddings_connector import (
    AudioEmbeddings1DConnectorConfigurator,
    Embeddings1DConnectorConfigurator,
)
from vforge.models.ltx.embeddings_processor import EmbeddingsProcessor
from vforge.models.ltx.feature_extractor import FeatureExtractorV2


def checkpoint_config(path):
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        raw = (handle.metadata() or {}).get("config")
    if raw is None:
        raise ValueError("base LTX checkpoint must contain config metadata")
    return json.loads(raw)


def load_connectors(base_path, *, device="cpu", dtype=torch.bfloat16):
    config = checkpoint_config(base_path)
    # Independent DMD text preprocessing has always run without autocast.
    ops = AttentionOps(norm_dtype=None)
    ops_config = vforge_ops.get_ops_config()
    if ops_config is not None and ops_config.backend == "sdpa_math":
        # Masked SDPA calls otherwise use PyTorch's default backend priority,
        # independently of the configured unmasked attention backend.
        ops = AttentionOps(
            norm_dtype=None,
            masked_attention_function=partial(
                vforge_ops.sdpa_attention, priority=(torch.nn.attention.SDPBackend.MATH,)
            ),
        )
    with torch.device("meta"):
        processor = EmbeddingsProcessor(
            video_connector=Embeddings1DConnectorConfigurator.from_config(config, ops=ops),
            audio_connector=AudioEmbeddings1DConnectorConfigurator.from_config(config, ops=ops),
        )
    state = {}
    with safe_open(str(base_path), framework="pt", device="cpu") as handle:
        for name in processor.state_dict():
            prefix, suffix = name.split(".", 1)
            source = prefix.replace("_connector", "_embeddings_connector")
            key = f"model.diffusion_model.{source}.{suffix}"
            state[name] = handle.get_tensor(key).to(device=device, dtype=dtype)
    processor.load_state_dict(state, strict=True, assign=True)
    return processor.eval().requires_grad_(False)


@torch.no_grad()
def encode_prompts(base_path, gemma_path, prompts, *, device, max_length=1024):
    """Encode prompts sequentially with one Gemma load; retain only CPU contexts."""
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer

    config = AutoConfig.from_pretrained(gemma_path, local_files_only=True)
    if config.model_type != "gemma3":
        raise ValueError("LTX-2.3 prompt encoding requires a local Gemma 3 directory")
    tokenizer = AutoTokenizer.from_pretrained(gemma_path, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = (
        AutoModelForImageTextToText.from_pretrained(
            gemma_path,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    processor = load_connectors(base_path, device=device)
    try:
        cfg = checkpoint_config(base_path)["transformer"]
        with torch.device("meta"):
            extractor = FeatureExtractorV2(
                config.text_config.hidden_size,
                config.text_config.num_hidden_layers,
                cfg["num_attention_heads"] * cfg["attention_head_dim"],
                cfg["audio_num_attention_heads"] * cfg["audio_attention_head_dim"],
            )
        with safe_open(str(base_path), framework="pt", device="cpu") as handle:
            state = {
                name: handle.get_tensor(f"text_embedding_projection.{name}").to(
                    device=device,
                    dtype=torch.bfloat16,
                )
                for name in extractor.state_dict()
            }
        extractor.load_state_dict(state, strict=True, assign=True)
        processor.feature_extractor = extractor.eval().requires_grad_(False)
        results = []
        for prompt in prompts:
            ids = tokenizer(prompt.strip(), truncation=True, max_length=max_length)["input_ids"]
            if tokenizer.bos_token_id is None:
                raise ValueError("Gemma tokenizer requires a BOS token")
            if not ids or ids[0] != tokenizer.bos_token_id:
                ids = [tokenizer.bos_token_id, *ids][:max_length]
            tokens = tokenizer.pad(
                {"input_ids": [ids]},
                padding="max_length",
                max_length=max_length,
                return_attention_mask=True,
                return_tensors="pt",
            ).to(device)
            hidden = model.model(**tokens, output_hidden_states=True).hidden_states
            result = processor.process_hidden_states(hidden, tokens["attention_mask"])
            results.append(
                {
                    "video_context": result.video_encoding.cpu(),
                    "audio_context": result.audio_encoding.cpu(),
                }
            )
            del hidden, result
        return results
    finally:
        del model, processor
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()


def encode_prompt(base_path, gemma_path, prompt, *, device, max_length=1024):
    return encode_prompts(base_path, gemma_path, [prompt], device=device, max_length=max_length)[0]


# Training retains its established import and identical encoding behavior.
encode_negative_prompt = encode_prompt
