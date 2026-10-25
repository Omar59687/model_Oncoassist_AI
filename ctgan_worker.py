"""Isolated SDV/PyTorch process; never imports TensorFlow or validation data."""

import argparse
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


def train_and_sample(real, needed, epochs, batch_size, pac, seed):
    import torch
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import CTGANSynthesizer

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    frame = pd.DataFrame(real, columns=[f"latent_{i}" for i in range(real.shape[1])])
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=frame)
    synthesizer = CTGANSynthesizer(
        metadata=metadata, epochs=epochs, batch_size=batch_size,
        pac=pac, enable_gpu=False, verbose=False,
    )
    synthesizer.fit(frame)
    return synthesizer.sample(num_rows=needed).to_numpy(dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--pac", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.diagnose:
        dummy = np.random.default_rng(42).normal(size=(86, 40))
        try:
            train_and_sample(dummy, 1, 1, 86, 10, 42)
        except Exception as exc:
            print(json.dumps({"input": "86x40 deterministic dummy data",
                              "old_batch_size": 86, "pac": 10,
                              "exception_type": type(exc).__name__,
                              "traceback": traceback.format_exc(limit=15)}))
        else:
            print(json.dumps({"input": "86x40 deterministic dummy data",
                              "old_batch_size": 86, "pac": 10,
                              "exception_type": None}))
        return
    if any(x is None for x in (args.input, args.output, args.epochs, args.batch_size,
                               args.pac, args.seed)):
        parser.error("Generation requires input, output, epochs, batch-size, pac and seed")
    with np.load(args.input, allow_pickle=False) as data:
        real = data["minority"]
        needed = int(data["needed"])
    if real.ndim != 2 or needed <= 0 or args.batch_size % 2 or args.batch_size % args.pac:
        raise ValueError("Invalid minority data or CTGAN batch configuration")
    generated = train_and_sample(real, needed, args.epochs, args.batch_size,
                                 args.pac, args.seed)
    np.save(args.output, generated, allow_pickle=False)


if __name__ == "__main__":
    main()
