"""Save and restore the full training state. Design: DESIGN.md.

    meta = {"config": config_sha, "tokenizer": tokenizer_sha, "embedding_policy_id": ...}
    step, extra = load(root, model, optimizer, scheduler, meta) or (0, {})
    for step in range(step, total):
        ...train one step...
        if (step + 1) % every == 0:
            save(root, step + 1, model, optimizer, scheduler, extra, meta=meta)

Works with plain, DDP and FSDP2 models, on GPU or CPU, with or without a process group.
Self-test (CPU, a few seconds): python ckpt.py
"""
import hashlib
import json
import random
import shutil
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict


def _rank():
    return dist.get_rank() if dist.is_initialized() else 0


def _barrier():
    if dist.is_initialized():
        dist.barrier()


def _rng_state():
    s = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state()
    return s


def _set_rng_state(s):
    random.setstate(s["python"])
    torch.set_rng_state(s["torch"])
    if "cuda" in s and torch.cuda.is_available():
        torch.cuda.set_rng_state(s["cuda"])


def _frozen_hash(model):
    """sha256 over every parameter that does not train. A frozen part must never change (Session 15)."""
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        if not p.requires_grad:
            t = p.full_tensor() if hasattr(p, "full_tensor") else p  # DTensor: gather the shards
            h.update(name.encode())
            h.update(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _meta(path):
    return json.loads((path / "meta.json").read_text())


def save(root, step, model, optimizer, scheduler=None, extra=None, keep=3, keep_every=0, meta=None):
    """Save the state after `step` training steps into root/step_<step>. Call on every rank.

    Keeps the newest `keep` checkpoints, and every one whose step is a multiple of `keep_every`.
    `meta` (plain JSON: config hash, tokenizer hash, embedding policy) is stored for load() to check.
    Raises if a frozen parameter changed since the previous checkpoint.
    """
    root = Path(root)
    final = root / f"step_{step:08d}"
    if (final / "COMPLETE").exists():
        return
    frozen = _frozen_hash(model)
    prev = latest(root)
    if prev and _meta(prev)["frozen"] != frozen:
        raise RuntimeError(f"frozen parameters changed since {prev.name}; not saving")
    tmp = final.with_suffix(".tmp")
    if _rank() == 0:
        shutil.rmtree(tmp, ignore_errors=True)  # left by a save that crashed
        shutil.rmtree(final, ignore_errors=True)  # no COMPLETE: a half-done download
        tmp.mkdir(parents=True)
    _barrier()

    model_sd, optim_sd = get_state_dict(model, optimizer)
    dcp.save({"model": model_sd, "optim": optim_sd}, checkpoint_id=tmp)
    torch.save({"step": step, "extra": extra or {}, "rng": _rng_state(),
                "scheduler": scheduler.state_dict() if scheduler else None},
               tmp / f"rank{_rank()}.pt")
    _barrier()

    # Commit: COMPLETE is written last, so only a checkpoint with COMPLETE is ever read.
    # The same rule holds on S3, which has no rename: upload COMPLETE after every other file.
    if _rank() == 0:
        (tmp / "meta.json").write_text(json.dumps({"step": step, "meta": meta, "frozen": frozen}, indent=1))
        (tmp / "COMPLETE").write_text(f"{step}\n")
        tmp.rename(final)
        done = sorted(root.glob("step_*[0-9]"))
        for old in done[:-keep]:
            if not (keep_every and int(old.name[5:]) % keep_every == 0):
                shutil.rmtree(old)
    _barrier()


def latest(root):
    """Newest COMPLETE checkpoint folder in root, or None. Restart scripts and watchers use this too."""
    done = sorted(p.parent for p in Path(root).glob("step_*[0-9]/COMPLETE"))
    return done[-1] if done else None


def load(root, model, optimizer, scheduler=None, meta=None):
    """Restore the newest COMPLETE checkpoint in root, RNGs included. Call on every rank.

    Returns (step, extra), or None if root has no COMPLETE checkpoint.
    The model may sit on another device or world size than at save time (DCP reshards).
    Strict, because a misload looks like a normal run (Session 15): raises if `meta` differs from
    the saved one, or if a model or optimizer key is missing from the checkpoint (a renamed layer).
    """
    path = latest(root)
    if path is None:
        return None
    saved = _meta(path)["meta"] or {}
    if meta is not None and saved != meta:
        diff = sorted(k for k in {*saved, *meta} if saved.get(k) != meta.get(k))
        raise ValueError(f"{path.name} was saved with other {diff}: {saved} != {meta}")

    model_sd, optim_sd = get_state_dict(model, optimizer)
    sd = {"model": model_sd, "optim": optim_sd}
    dcp.load(sd, checkpoint_id=path)
    set_state_dict(model, optimizer, model_state_dict=sd["model"], optim_state_dict=sd["optim"])

    own = path / f"rank{_rank()}.pt"
    state = torch.load(own if own.exists() else path / "rank0.pt", weights_only=True)  # world size grew
    if scheduler:
        scheduler.load_state_dict(state["scheduler"])
    _set_rng_state(state["rng"])
    return state["step"], state["extra"]


def _selftest():
    import tempfile

    def make(seed):
        torch.manual_seed(seed)
        model = torch.nn.Sequential(torch.nn.Linear(16, 64), torch.nn.GELU(), torch.nn.Dropout(0.2),
                                    torch.nn.Linear(64, 1))
        model[0].bias.requires_grad_(False)  # a frozen part, as the Kronecker codec is
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.5)
        return model, opt, sched

    def train(model, opt, sched, steps, root=None):
        losses = []
        for step in steps:
            g = torch.Generator().manual_seed(step)  # the batch is a function of the step
            x, y = torch.randn(8, 16, generator=g), torch.randn(8, 1, generator=g)
            loss = torch.nn.functional.mse_loss(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
            if root:
                save(root, step + 1, model, opt, sched, {"tokens": (step + 1) * 8}, keep=3, keep_every=2, meta=META)
        return losses

    META = {"config": "sha-a", "tokenizer": "sha-t"}
    straight = train(*make(0), range(10))
    with tempfile.TemporaryDirectory() as root:
        root = Path(root)
        first = train(*make(0), range(5), root)
        (root / "step_00000009.tmp").mkdir()  # a save that crashed on disk
        (root / "step_00000010").mkdir()  # a half-done S3 download, no COMPLETE
        model, opt, sched = make(123)  # other weights and RNG; load must replace all of it
        step, extra = load(root, model, opt, sched, META)
        assert (step, extra) == (5, {"tokens": 40}), (step, extra)
        resumed = first + train(model, opt, sched, range(step, 10))
        kept = sorted(p.parent.name for p in root.glob("step_*[0-9]/COMPLETE"))
        assert kept == ["step_00000002", "step_00000003", "step_00000004", "step_00000005"], kept

        # Silent-failure guards (Session 15). Each must raise, not load or save quietly.
        def raises(f, *a, **k):
            try:
                f(*a, **k)
            except (Exception, dcp.api.CheckpointException):  # the latter is a BaseException
                return True
            return False
        assert raises(load, root, *make(0), meta={**META, "config": "sha-b"}), "other config loaded"
        renamed = torch.nn.Sequential(make(0)[0])  # every key gains a "0." prefix: a converter's renaming
        assert raises(load, root, renamed, torch.optim.AdamW(renamed.parameters())), "renamed layers loaded"
        with torch.no_grad():
            model[0].bias.add_(1.0)  # a frozen part was trained by mistake
        assert raises(save, root, 11, model, opt, sched, meta=META), "changed frozen part saved"
        assert not (root / "step_00000011.tmp").exists()
    assert load(root, *make(0)) is None
    assert resumed == straight, f"losses differ after restore:\n{straight}\n{resumed}"
    print(f"selftest ok: {len(straight)} losses bit-equal after a CPU restore at step 5")


if __name__ == "__main__":
    _selftest()
