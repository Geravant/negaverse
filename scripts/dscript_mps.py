"""Run the D-SCRIPT CLI on Apple Metal (MPS).

D-SCRIPT is CUDA-hardcoded: it gates every `.cuda()` on
`use_cuda = (device > -1) and torch.cuda.is_available()`. On a Mac that's always
False → CPU. This launcher fakes `torch.cuda.is_available()` and routes `.cuda()`
to `.to('mps')`, so passing `-d 0` runs the model on Metal. PYTORCH_ENABLE_MPS_
FALLBACK routes any op MPS lacks back to CPU instead of crashing.

    PYTORCH_ENABLE_MPS_FALLBACK=1 .venv-dscript/bin/python scripts/dscript_mps.py \
        train --train ... --test ... --embedding ... -d 0
"""
import os
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def _patch_dscript_source():
    """MPS autograd rejects .view()/backward on non-contiguous tensors produced by
    D-SCRIPT's transposes. Make the two hot spots contiguous, in place, idempotently
    — so a fresh `.venv-dscript` works without hand-editing the package."""
    import dscript
    root = os.path.dirname(dscript.__file__)
    edits = {
        os.path.join(root, "models", "embedding.py"): [
            ("z = self.proj(h.view(-1, h.size(2)))",
             "z = self.proj(h.reshape(-1, h.size(2)).contiguous())"),
            ("z = z.view(x.size(0), x.size(1), -1)",
             "z = z.reshape(x.size(0), x.size(1), -1).contiguous()")],
        os.path.join(root, "models", "contact.py"): [
            ("z0 = z0.transpose(1, 2)\n", "z0 = z0.transpose(1, 2).contiguous()\n"),
            ("z1 = z1.transpose(1, 2)\n", "z1 = z1.transpose(1, 2).contiguous()\n"),
            ("z_cat = torch.cat([z_dif, z_mul], 1)\n",
             "z_cat = torch.cat([z_dif, z_mul], 1).contiguous()\n")],
    }
    for path, subs in edits.items():
        if not os.path.exists(path):
            continue
        src = open(path).read()
        for old, new in subs:
            if old in src and new not in src:
                src = src.replace(old, new)
        open(path, "w").write(src)


_patch_dscript_source()

import torch

if not torch.backends.mps.is_available():
    sys.exit("MPS not available on this machine")

_MPS = torch.device("mps")
torch.cuda.is_available = lambda: True
torch.cuda.set_device = lambda *a, **k: None
torch.cuda.get_device_name = lambda *a, **k: "Apple MPS"
torch.cuda.device_count = lambda: 1
torch.cuda.empty_cache = lambda *a, **k: None
torch.cuda.synchronize = lambda *a, **k: None


def _to_mps(self, *a, **k):
    return self.to(_MPS)


torch.Tensor.cuda = _to_mps
torch.nn.Module.cuda = _to_mps

# MPS autograd rejects .view() on non-contiguous tensors ("Use .reshape(...)").
# reshape is view-when-possible, copy-otherwise → always safe. Alias it so
# D-SCRIPT's model runs on Metal without editing its source.
_orig_view = torch.Tensor.view
torch.Tensor.view = lambda self, *a, **k: self.reshape(*a, **k)

from dscript.__main__ import main   # noqa: E402

sys.argv = ["dscript"] + sys.argv[1:]
main()
