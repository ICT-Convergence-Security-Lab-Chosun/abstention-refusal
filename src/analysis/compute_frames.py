#!/usr/bin/env python3
"""Response-frame composition (Figure 4b / §4.2) from frozen generations, no plotting.

For each headline model and each direction (comp1 = v_abs, refusal_dir = v_ref), the share of
valid (+alpha) TriviaQA responses that are epistemic-framed / safety-framed / other, using the
rule-based keyword classifier. Writes runs/frames.json and prints the shares quoted in §4.2
("epistemic-framed 0.87-0.96 ... almost entirely safety-framed 0.96-1.00").

Run:  python3 src/analysis/compute_frames.py
"""
import os, json
from frames_common import style_frame, MODELS

PKG = os.environ.get("ABST_PKG") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = f"{PKG}/runs/frames.json"


def main():
    frames = style_frame()
    json.dump(frames, open(OUT, "w"), indent=1)
    print("saved", OUT)
    print(f"{'model':8} {'direction':12} {'epistemic':>10} {'safety':>8} {'other':>8}")
    for name in [m[0] for m in MODELS]:
        for arm, lab in [("comp1", "v_abs"), ("refusal_dir", "v_ref")]:
            c = frames[name][arm]
            print(f"{name:8} {lab:12} {c['epistemic']:>10.2f} {c['safety']:>8.2f} {c['other']:>8.2f}")


if __name__ == "__main__":
    main()
