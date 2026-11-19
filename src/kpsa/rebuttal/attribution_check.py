"""Check that the benchmark's attribution path computes the evaluated attribution.

Runs vision_cost_bench's `attribution` method (fp32) on its 32 benchmark images
(offset 49 of the 200 classes, i.e. the first 32 query49 images) and compares
its top-37 features with the top-37 of the cached activation attribution used by
the accuracy evaluation (vision_cache.pt, query49_attribution).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from .vision_cost_bench import build, load_images


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    cache = torch.load(Path.home() / "main/data/kpsa-rebuttal/vision_cache.pt", weights_only=False)
    pixels, labels = load_images(32, "vit")
    classes = torch.linspace(0, 999, 200).round().long()
    assert torch.equal(cache["roles"]["query49"][:32], classes[:32] * 50 + 49), "image sets differ"
    ref = torch.topk(cache["query49_attribution"][:32].float(), 37, dim=1).indices
    out = {"images": 32}
    for method in sys.argv[2:] or ["attribution"]:
        for prec in ("fp32", "bf16"):
            run, _ = build(method, prec, 6400, 1, 37)
            with torch.enable_grad():
                got = run(pixels.cuda(), labels.cuda()).cpu()
            overlap = [len(set(a.tolist()) & set(b.tolist())) / 37 for a, b in zip(got, ref)]
            out[f"{method}_{prec}"] = {"top37_overlap_mean": sum(overlap) / 32, "top37_overlap_min": min(overlap)}
    print(json.dumps(out))
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
