"""Verify real saved checkpoints and short CPU inference without downloading."""
from pathlib import Path
import argparse
import gc
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stem_studio.assets import CATALOG, valid
from stem_studio.engine import construct_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pack', type=Path)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(4)
    for key, spec in CATALOG.items():
        path = args.pack / spec['name']
        if not valid(path, spec):
            raise ValueError('Checkpoint verification failed: ' + str(path))
        began = time.monotonic()
        model = construct_model(spec, path)
        hop = spec['config']['model']['stft_hop_length']
        frames = (44100 // hop) * hop
        with torch.inference_mode():
            output = model(torch.randn(1, 2, frames) * .03)
        expected = (1, 6, 2, frames) if key == 'sw6' else (1, 2, frames)
        if tuple(output.shape) != expected or not torch.isfinite(output).all():
            raise ValueError('Unexpected inference output: ' + key)
        print(key, 'strict loading + finite CPU inference:', expected,
              f'({time.monotonic() - began:.2f} s)', flush=True)
        del model, output
        gc.collect()


if __name__ == '__main__':
    main()
