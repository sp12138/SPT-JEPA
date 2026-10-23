"""Segment-chain DMD: one resumable sample and one carry update per G/C step."""

import logging
from dataclasses import asdict, replace

import torch

from vforge.algorithms.dmd.blockwise import (
    BlockDistributionMatchingAlgorithm,
    BlockDistributionMatchingRuntime,
    BlockDistributionMatchingSteps,
)
from vforge.algorithms.dmd.config import SegmentChainDistributionMatchingConfig
from vforge.algorithms.dmd.model import DistributionMatchingData
from vforge.data.multi_stream_loader import MultiStreamStatefulDataLoader


def _snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, dict):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_snapshot(item) for item in value)
    return value


class SegmentChainDataLoader:
    """Advance only after critic update; checkpoint committed CPU state, never KV."""

    _vforge_resumable_dataloader = True
    _vforge_bit_exact_resume = True

    def __init__(self, loader, config, adapter):
        self.source = MultiStreamStatefulDataLoader({"sample": loader})
        self.config = config
        self.adapter = adapter
        self.sample = None
        self.segment = 0
        self.carry = None
        self.pending = None
        self.in_flight = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.in_flight:
            raise RuntimeError("segment must be committed before requesting the next batch")
        if self.sample is None or self.segment == self.config.segments_per_sample:
            self.sample = next(self.source)["sample"]
            self.segment, self.carry = 0, None
        self.in_flight = True
        return self.sample

    def prepare(self, prepared):
        return self.adapter.prepare_segment(prepared, self.carry)

    def retain(self, generated, prepared, phase):
        if not self.in_flight:
            raise RuntimeError("rollout requires an active segment")
        # G's pre-update output wins on G+C steps. C must use the same old carry.
        if self.pending is None or phase == "generator":
            self.pending = ({k: v.detach() for k, v in generated.items()}, prepared)

    def commit(self, _root, _step):
        if not self.in_flight or self.pending is None:
            raise RuntimeError("no completed segment rollout to commit")
        generated, prepared = self.pending
        carry = self.adapter.advance_segment(generated, prepared, self.carry)
        self.carry = _snapshot(carry)
        self.segment += 1
        self.pending = None
        self.in_flight = False

    def state_dict(self):
        if self.in_flight or self.pending is not None:
            raise RuntimeError("checkpoint requires a fully committed G/C step")
        return _snapshot(
            {
                "version": 1,
                "source": self.source.state_dict(),
                "sample": self.sample,
                "segment": self.segment,
                "carry": self.carry,
                "chain": asdict(self.config),
            }
        )

    def load_state_dict(self, state):
        if state.get("version") != 1 or state.get("chain") != asdict(self.config):
            raise ValueError("incompatible segment chain checkpoint")
        segment = int(state["segment"])
        if not 0 <= segment <= self.config.segments_per_sample:
            raise ValueError("invalid segment cursor")
        if segment and (state["sample"] is None or state["carry"] is None):
            raise ValueError("segment checkpoint is missing sample or carry")
        self.source.load_state_dict(state["source"])
        self.sample, self.carry = _snapshot(state["sample"]), _snapshot(state["carry"])
        self.segment, self.pending, self.in_flight = segment, None, False


def build_segment_chain_data(family, *, config, seed, stream_seed_offsets):
    logging.getLogger(__name__).info(
        "Segment chain: %s slots, %s segments, %s actual blocks (requested %s)",
        config.chain.segment_slots,
        config.chain.segments_per_sample,
        config.chain.actual_rollout_blocks,
        config.chain.max_rollout_blocks,
    )
    source, contract = family.build_train_stream(0)
    loader = SegmentChainDataLoader(source, config.chain, family.generator_adapter)
    runtime = BlockDistributionMatchingRuntime(family=family, config=config, seed=seed)
    return DistributionMatchingData(
        dataloader=loader,
        prepare_batch=lambda batch, device: loader.prepare(family.prepare_batch(batch, device)),
        rollout=runtime.rollout,
        rollout_trace=runtime.rollout_trace,
        sample_score_noise=runtime.sample_score_noise,
        valid_masks=family.valid_masks,
        loss_partitions=family.loss_partitions,
        batch_selector=lambda batch, _phase: batch,
        workload=family.workload,
        metrics=family.metrics,
        dataloader_resume_contract={
            "type": "segment_chain_dmd",
            "source": contract,
            "chain": asdict(config.chain),
            "advance": "generator_else_critic_after_step",
            "assets": family.generator_adapter.chain_asset_contract(),
        },
    )


class SegmentChainDistributionMatchingAlgorithm(BlockDistributionMatchingAlgorithm):
    name = "segment_chain_distribution_matching"
    config_type = SegmentChainDistributionMatchingConfig
    data_builder = staticmethod(build_segment_chain_data)

    @staticmethod
    def create_steps(**kwargs):
        return SegmentChainDistributionMatchingSteps(**kwargs)

    @staticmethod
    def _build_phases(request, *, config, steps):
        phases = BlockDistributionMatchingAlgorithm._build_phases(
            request, config=config, steps=steps
        )
        phases["critic"] = replace(phases["critic"], post_update_hook=steps.data.dataloader.commit)
        return phases


class SegmentChainDistributionMatchingSteps(BlockDistributionMatchingSteps):
    def retain_rollout(self, generated, prepared, phase):
        self.data.dataloader.retain(generated, prepared, phase)

    def score_gradient(self, noisy, generated, level, prepared, **scores):
        # The reference reduces the contiguous noisy region, without inserting
        # masked zeros into the reduction. Audio handover stays in this region.
        masks = self.data.valid_masks(prepared)
        offsets = {}
        for key in generated:
            mask = None if masks is None else masks.get(key)
            offset = 0 if mask is None else int((~mask[0, :, 0]).sum())
            if mask is not None and (mask[:, :offset].any() or not mask[:, offset:].all()):
                raise ValueError("segment score mask must select a contiguous suffix")
            offsets[key] = offset

        def project(values):
            return {key: value[:, offsets[key] :] for key, value in values.items()}

        def wrap(score):
            return lambda _values, active_level: project(score(noisy, active_level))

        gradients = self._dmd_score_gradient_bundle(
            project(noisy),
            project(generated),
            level,
            **{name: wrap(score) for name, score in scores.items()},
            component_guidance_scales=self.config.score.component_guidance_scales,
        )
        return {
            key: torch.cat([torch.zeros_like(generated[key][:, : offsets[key]]), value], dim=1)
            for key, value in gradients.items()
        }
