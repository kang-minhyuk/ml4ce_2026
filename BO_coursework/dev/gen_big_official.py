"""Write 400 fresh official-style scenarios (ids 10680-11079, seed 2027) for a larger paired-comparison split
'off_big' (analysis only; never used for the library)."""
import json, numpy as np
from gen_official import make, practice_ranges, OUT
ranges, fixed = practice_ranges(); rng = np.random.default_rng(2027)
for i in range(680, 1080):
    (OUT / f"scenario_{i:04d}.json").write_text(json.dumps(make(i, rng, ranges, fixed), indent=1))
S = json.load(open("splits.json")); S["off_big"] = list(range(10680, 11080))
json.dump(S, open("splits.json", "w"))
print("ok")
