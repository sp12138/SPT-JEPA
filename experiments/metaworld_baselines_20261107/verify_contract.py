"""Compatibility entry point for the corrected native-window data contract.

The old common-window8-prefix contract is archived with the immutable pilot
runtime. It is not the formal native-coverage protocol.
"""
import argparse

from experiments.metaworld_baselines_20261107.verify_native_coverage import main


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    main(args.output)
