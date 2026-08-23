"""Train NCA (Growing Neural Cellular Automata) to grow a procedurally
generated RGBA target pattern from a single seed cell.

    python models/nca/example.py --device auto --epochs 2000

See model.py for the perceive/update/alive-masking mechanism and
papers/README.md for the reference. Unlike every other model in this
repo, there is no classification/segmentation dataset here: the "training
data" is a single procedurally generated target image (see
`model.make_target`), and each training iteration is one freshly-seeded,
randomly-unrolled growth trajectory compared against it.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cnn_playground.device import add_device_arg, resolve_device  # noqa: E402
from cnn_playground.utils.seed import set_seed  # noqa: E402
from model import NCAModel, make_target, seed_state  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=2000, help="training iterations (each = one unrolled grow trajectory)")
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--grid-size", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--min-steps", type=int, default=48)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    add_device_arg(parser)
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    target = make_target(args.grid_size).to(device)  # (4, H, W)
    target_batch = target.unsqueeze(0).expand(args.batch_size, -1, -1, -1)

    model = NCAModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    t0 = time.perf_counter()
    last_loss = None
    for it in range(1, args.epochs + 1):
        state = seed_state(args.batch_size, args.grid_size, device=device)
        n_steps = torch.randint(args.min_steps, args.max_steps + 1, (1,)).item()

        opt.zero_grad()
        final_state = model(state, n_steps)
        loss = F.mse_loss(final_state[:, :4], target_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last_loss = loss.item()

        if it == 1 or it % args.log_every == 0 or it == args.epochs:
            print(f"iter {it:5d} | loss {last_loss:.5f}")
    train_time = time.perf_counter() - t0

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"RESULT: model=nca metric_name=final_mse metric={last_loss:.5f} "
        f"params={n_params} train_time_s={train_time:.2f}"
    )


if __name__ == "__main__":
    main()
