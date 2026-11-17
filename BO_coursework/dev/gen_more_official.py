"""Write extra official-style scenarios (ids 10380+) for the glutamine-response library (analysis only)."""
import json, numpy as np
from gen_official import make, practice_ranges, OUT
ranges, fixed = practice_ranges(); rng = np.random.default_rng(2026)
for i in range(380, 680):
    (OUT / f"scenario_{i:04d}.json").write_text(json.dumps(make(i, rng, ranges, fixed), indent=1))
print("ok")
