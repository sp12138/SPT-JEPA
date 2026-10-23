"""Hydra configuration package for vforge command-line entries."""

from hydra.core.config_store import ConfigStore

from vforge.hydra_config import StandaloneConfig, to_plain

# Load defaults before CLI overrides, using the same source as config_from_dict.
# Plain nodes keep frozen dataclasses mergeable and model extras extensible;
# config_from_dict still owns conversion and validation after composition.
ConfigStore.instance().store(
    name="native_defaults", node=to_plain(StandaloneConfig()), provider="vforge"
)
