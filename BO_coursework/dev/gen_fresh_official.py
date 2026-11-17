"""Write 400 fresh official-style scenarios (ids 11080-11479, seed 2028) as split 'off_fresh': a final, untouched
validation split (never used for selection or the library). Analysis only."""
import json, numpy as np
from gen_official import make, practice_ranges, OUT
ranges, fixed = practice_ranges(); rng = np.random.default_rng(2028)
for i in range(1080, 1480):
    assert not (OUT / f"scenario_{i:04d}.json").exists()
    (OUT / f"scenario_{i:04d}.json").write_text(json.dumps(make(i, rng, ranges, fixed), indent=1))
S = json.load(open("splits.json")); S["off_fresh"] = list(range(11080, 11480))
json.dump(S, open("splits.json", "w"))
print("ok")
