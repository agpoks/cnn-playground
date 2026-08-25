"""Apples-to-apples benchmark: resnet (reference baseline) vs. the four new
CIFAR-10 classifiers added this session (odenet, liquidode, legendrekan,
obfconv), plus a separate report for nca (a different task, not comparable
on accuracy).

    python benchmarks/compare_new_models.py --device cpu --epochs 3

For every Group-A model: same CIFAR-10 data, same batch size, same fixed
epoch budget, each model's own already-tuned default learning rate. Reports
params, on-disk state_dict size, multiply-accumulates (MACs) per single-image
forward pass (hand-rolled hook counter over Conv2d/Linear only -- no
external FLOP-counting library), wall-clock training time, final test
accuracy, and CPU inference latency at batch=1 and batch=64.

Group B (nca) gets its own small section: params, disk size, MACs for one
CA step, training time for a fixed iteration budget, final MSE, and
inference latency for one step and for a full n_steps=48 unroll.

These are NOT converged/SOTA numbers -- a short, fixed, identical budget
across models, useful for *relative* comparison (which architecture is
cheaper/faster/more sample-efficient under equal compute), not an absolute
accuracy benchmark. See docs/source/benchmark_results.md for the write-up.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from cnn_playground.data import load_cifar10  # noqa: E402
from cnn_playground.device import add_device_arg, resolve_device  # noqa: E402
from cnn_playground.utils.seed import set_seed  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "models" / "resnet"))
sys.path.insert(0, str(REPO_ROOT / "models" / "odenet"))
sys.path.insert(0, str(REPO_ROOT / "models" / "liquidode"))
sys.path.insert(0, str(REPO_ROOT / "models" / "legendrekan"))
sys.path.insert(0, str(REPO_ROOT / "models" / "obfconv"))
sys.path.insert(0, str(REPO_ROOT / "models" / "nca"))


# Import each model.py under its own module name to avoid collisions (they
# all define a top-level `model` module otherwise).
def _import_from(path, module_name):
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


resnet_mod = _import_from(REPO_ROOT / "models" / "resnet" / "model.py", "bench_resnet")
odenet_mod = _import_from(REPO_ROOT / "models" / "odenet" / "model.py", "bench_odenet")
liquidode_mod = _import_from(REPO_ROOT / "models" / "liquidode" / "model.py", "bench_liquidode")
legendrekan_mod = _import_from(REPO_ROOT / "models" / "legendrekan" / "model.py", "bench_legendrekan")
obfconv_mod = _import_from(REPO_ROOT / "models" / "obfconv" / "model.py", "bench_obfconv")
nca_mod = _import_from(REPO_ROOT / "models" / "nca" / "model.py", "bench_nca")

GROUP_A = [
    ("resnet", resnet_mod.ResNetModel, {}, 1e-3),
    ("odenet", odenet_mod.ODENetModel, {}, 1e-3),
    ("liquidode", liquidode_mod.LiquidODENetModel, {}, 1e-3),
    ("legendrekan", legendrekan_mod.LegendreKANModel, {}, 1e-3),
    ("obfconv", obfconv_mod.OBFConvModel, {"basis": "laguerre"}, 1e-3),
]


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def disk_size_mb(model: nn.Module) -> float:
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    torch.save(model.state_dict(), path)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    os.remove(path)
    return size_mb


def count_macs(model: nn.Module, input_shape) -> float:
    """Hand-rolled MAC counter via forward hooks on Conv2d/Linear only
    (ignores elementwise ops, norm layers, etc. -- standard practice for
    this kind of relative comparison)."""
    total = [0]
    hooks = []

    def conv_hook(mod, inp, out):
        out_numel = out.numel() / out.shape[0]  # per-sample
        k = mod.kernel_size
        macs = out_numel * (mod.in_channels / mod.groups) * k[0] * k[1]
        total[0] += macs

    def linear_hook(mod, inp, out):
        total[0] += mod.in_features * mod.out_features

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    model.eval()
    with torch.no_grad():
        model(torch.randn(1, *input_shape))
    for h in hooks:
        h.remove()
    return total[0] / 1e6  # MMACs


def measure_inference_latency(model: nn.Module, input_shape, batch_sizes=(1, 64), n_runs=50, n_warmup=10):
    model.eval()
    results = {}
    for bs in batch_sizes:
        x = torch.randn(bs, *input_shape)
        with torch.no_grad():
            for _ in range(n_warmup):
                model(x)
            t0 = time.perf_counter()
            for _ in range(n_runs):
                model(x)
            elapsed = time.perf_counter() - t0
        results[bs] = (elapsed / n_runs) * 1000  # ms per forward pass
    return results


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
    return correct / total


def run_group_a(device, epochs, batch_size, seed):
    set_seed(seed)
    train_ds = load_cifar10(train=True)
    test_ds = load_cifar10(train=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    rows = []
    for name, cls, kwargs, lr in GROUP_A:
        set_seed(seed)
        model = cls(**kwargs).to(device)
        n_params = count_params(model)
        size_mb = disk_size_mb(model)
        macs = count_macs(cls(**kwargs), (3, 32, 32))

        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        t0 = time.perf_counter()
        for epoch in range(1, epochs + 1):
            model.train()
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                opt.zero_grad()
                loss = loss_fn(model(imgs), labels)
                loss.backward()
                opt.step()
        train_time = time.perf_counter() - t0

        test_acc = evaluate(model, test_loader, device)
        latency = measure_inference_latency(model.cpu(), (3, 32, 32))

        row = {
            "model": name,
            "params": n_params,
            "size_mb": size_mb,
            "mmacs": macs,
            "train_time_s": train_time,
            "test_acc": test_acc,
            "lat_bs1_ms": latency[1],
            "lat_bs64_ms": latency[64],
        }
        rows.append(row)
        print(
            f"RESULT: model={name} metric_name=test_acc metric={test_acc:.4f} "
            f"params={n_params} size_mb={size_mb:.3f} mmacs={macs:.2f} "
            f"train_time_s={train_time:.2f} lat_bs1_ms={latency[1]:.3f} lat_bs64_ms={latency[64]:.3f}"
        )
    return rows


def run_group_b(device, iterations, seed):
    set_seed(seed)
    NCAModel = nca_mod.NCAModel
    make_target = nca_mod.make_target
    seed_state = nca_mod.seed_state

    grid_size, batch_size, min_steps, max_steps = 40, 8, 48, 64
    target = make_target(grid_size).to(device)
    target_batch = target.unsqueeze(0).expand(batch_size, -1, -1, -1)

    model = NCAModel().to(device)
    n_params = count_params(model)
    size_mb = disk_size_mb(model)

    # count_macs() assumes model(x); NCA's forward is model(state, n_steps), so measure
    # MACs for one step directly here instead of reusing that helper.
    x1 = seed_state(1, grid_size, device="cpu")
    total = [0]
    hooks = []

    def conv_hook(mod, inp, out):
        out_numel = out.numel() / out.shape[0]
        k = mod.kernel_size
        macs = out_numel * (mod.in_channels / mod.groups) * k[0] * k[1]
        total[0] += macs

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
    model.eval()
    with torch.no_grad():
        model(x1, 1)
    for h in hooks:
        h.remove()
    macs_one_step = total[0] / 1e6
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    t0 = time.perf_counter()
    last_loss = None
    for it in range(1, iterations + 1):
        state = seed_state(batch_size, grid_size, device=device)
        n_steps = torch.randint(min_steps, max_steps + 1, (1,)).item()
        opt.zero_grad()
        final_state = model(state, n_steps)
        loss = F.mse_loss(final_state[:, :4], target_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last_loss = loss.item()
    train_time = time.perf_counter() - t0

    model_cpu = model.cpu().eval()
    x1 = seed_state(1, grid_size, device="cpu")
    with torch.no_grad():
        for _ in range(10):
            model_cpu(x1, 1)
        t0 = time.perf_counter()
        for _ in range(50):
            model_cpu(x1, 1)
        lat_one_step_ms = (time.perf_counter() - t0) / 50 * 1000

        for _ in range(3):
            model_cpu(x1, 48)
        t0 = time.perf_counter()
        for _ in range(10):
            model_cpu(x1, 48)
        lat_full_unroll_ms = (time.perf_counter() - t0) / 10 * 1000

    row = {
        "model": "nca",
        "params": n_params,
        "size_mb": size_mb,
        "mmacs_one_step": macs_one_step,
        "train_time_s": train_time,
        "final_mse": last_loss,
        "iterations": iterations,
        "lat_one_step_ms": lat_one_step_ms,
        "lat_full_unroll_ms": lat_full_unroll_ms,
    }
    print(
        f"RESULT: model=nca metric_name=final_mse metric={last_loss:.5f} params={n_params} "
        f"size_mb={size_mb:.3f} mmacs_one_step={macs_one_step:.4f} train_time_s={train_time:.2f} "
        f"lat_one_step_ms={lat_one_step_ms:.4f} lat_full_unroll_ms={lat_full_unroll_ms:.3f}"
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3, help="fixed epoch budget for all Group-A models")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--nca-iterations", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    add_device_arg(parser)
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    print("\n=== Group A: CIFAR-10 classifiers ===")
    rows_a = run_group_a(device, args.epochs, args.batch_size, args.seed)

    print("\n=== Group B: NCA (not comparable to Group A) ===")
    row_b = run_group_b(device, args.nca_iterations, args.seed)

    print("\n=== Summary: Group A ===")
    header = f"{'model':<14}{'params':>10}{'size_mb':>10}{'mmacs':>10}{'train_s':>10}{'test_acc':>10}{'lat_bs1_ms':>12}{'lat_bs64_ms':>13}"
    print(header)
    for r in rows_a:
        print(
            f"{r['model']:<14}{r['params']:>10}{r['size_mb']:>10.3f}{r['mmacs']:>10.2f}"
            f"{r['train_time_s']:>10.2f}{r['test_acc']:>10.4f}{r['lat_bs1_ms']:>12.3f}{r['lat_bs64_ms']:>13.3f}"
        )

    print("\n=== Summary: Group B (NCA) ===")
    print(
        f"params={row_b['params']} size_mb={row_b['size_mb']:.3f} "
        f"mmacs_one_step={row_b['mmacs_one_step']:.4f} train_time_s={row_b['train_time_s']:.2f} "
        f"iterations={row_b['iterations']} final_mse={row_b['final_mse']:.5f} "
        f"lat_one_step_ms={row_b['lat_one_step_ms']:.4f} lat_full_unroll_ms={row_b['lat_full_unroll_ms']:.3f}"
    )


if __name__ == "__main__":
    main()
