"""Training-step throughput on CUDA: fp32 vs bf16 autocast vs channels_last vs torch.compile (batch 128, like training)."""
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import DEV, Arrays, Classifier, d4, load_meta, sync, to_input

m = load_meta()
A = Arrays()
BS = 128
idx = np.sort(np.random.RandomState(2).choice(len(m), BS * 8, replace=False))
raw_all = A.raw("dens", idx)
y = torch.randint(0, 2, (BS,), device=DEV)


def bench(name, amp=False, cl=False, compile_=False, steps=60):
    torch.manual_seed(0)
    net = Classifier("dens", 2).to(DEV)
    if cl:
        net = net.to(memory_format=torch.channels_last)
    model = torch.compile(net) if compile_ else net
    opt = torch.optim.AdamW(net.parameters(), 1e-3)
    t0 = None
    for it in range(steps + 10):
        if it == 10:
            sync(); t0 = time.time()
        k = it % 8
        raw = tuple(r[k * BS:(k + 1) * BS] for r in raw_all)
        x = to_input(d4(raw, "dens"), "dens")
        if cl:
            x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            loss = F.cross_entropy(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    sync()
    print(f"{name:32s} {steps * BS / (time.time() - t0):7.0f} img/s   loss {loss.item():.3f}", flush=True)


torch.backends.cudnn.benchmark = True
bench("fp32")
bench("fp32 + TF32", )
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
bench("TF32")
bench("bf16 autocast", amp=True)
bench("bf16 + channels_last", amp=True, cl=True)
try:
    bench("bf16 + channels_last + compile", amp=True, cl=True, compile_=True, steps=100)
except Exception as e:  # compile is optional
    print("compile failed:", type(e).__name__, str(e)[:200])
