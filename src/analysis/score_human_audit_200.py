#!/usr/bin/env python3
"""Score the 200-item human-audit sheets against the evaluators they check.

Reads the filled sheets in human_label_200/, joins each item to its automatic
verdict in KEY_200.json, and reports agreement, Cohen's kappa, both marginals and the confusion
matrix -- the columns of Table 6. The two validity sheets are one-sided by construction (the rule
assigns a single label per stratum), so kappa is undefined there and precision / miss rate are
reported instead. The frame sheet is three-class: accuracy, per-class recall and precision, and
multi-class kappa.

Run: python3 src/analysis/score_human_audit_200.py [--dir DIR]
"""
import os, re, json, math, argparse, collections

DIR_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "human_label_200")
YESNO = {"yes": True, "no": False}
FRAMES = ("epistemic", "safety", "other")


def wilson(k, n, z=1.96):
    if not n:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def kappa_binary(a, b, c, d):
    """a=both no, b=human only, c=auto only, d=both yes."""
    n = a + b + c + d
    if not n:
        return None
    po = (a + d) / n
    pe = ((a + b) / n) * ((a + c) / n) + ((c + d) / n) * ((b + d) / n)
    return (po - pe) / (1 - pe) if abs(1 - pe) > 1e-12 else None


def kappa_multi(pairs, labels):
    n = len(pairs)
    if not n:
        return None
    po = sum(h == m for h, m in pairs) / n
    hc = collections.Counter(h for h, _ in pairs)
    mc = collections.Counter(m for _, m in pairs)
    pe = sum((hc[l] / n) * (mc[l] / n) for l in labels)
    return (po - pe) / (1 - pe) if abs(1 - pe) > 1e-12 else None


def parse(path, tags):
    """{id: {tag: raw}} for every labelled item in a sheet."""
    body = open(path).read()
    out = {}
    for b in re.split(r"(?m)^(?=\d+\.\s+`[A-Z]\d+`)", body)[1:]:
        iid = re.match(r"\d+\.\s+`([A-Z]\d+)`", b).group(1)
        vals = {}
        for t in tags:
            # both hand-written forms occur: "`[TAG]:` yes" and "`[TAG]:yes"
            m = re.search(r"\[%s\]:`?[ \t]*([A-Za-z]+)" % t, b)
            vals[t] = m.group(1).lower() if m else None
        out[iid] = vals
    return out


def binary_report(title, pairs, pos_name="yes"):
    """pairs: list of (human_bool, auto_bool)."""
    a = b = c = d = 0
    for h, m in pairs:
        if not m and not h: a += 1
        elif m and not h: c += 1
        elif not m and h: b += 1
        else: d += 1
    n = a + b + c + d
    if not n:
        print(f"\n{title}: no labels"); return None
    agr = (a + d) / n
    k = kappa_binary(a, b, c, d)
    auto, hum = (c + d) / n, (b + d) / n
    lo, hi = wilson(a + d, n)
    print(f"\n{title}")
    print(f"  n={n}  agreement={agr:.4f} [95% CI {lo:.3f}, {hi:.3f}]  "
          f"kappa={'undefined' if k is None else f'{k:.4f}'}")
    print(f"  auto {pos_name}={auto:.4f}   human {pos_name}={hum:.4f}")
    print(f"  confusion  both-no={a}  auto-only={c}  human-only={b}  both-yes={d}")
    return {"n": n, "agreement": round(agr, 4), "kappa": (None if k is None else round(k, 4)),
            "auto": round(auto, 4), "human": round(hum, 4),
            "confusion": {"both_no": a, "auto_only": c, "human_only": b, "both_yes": d}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DIR_DEFAULT)
    args = ap.parse_args()
    D = args.dir
    K = json.load(open(os.path.join(D, "KEY_200.json")))
    items = K["items"]
    res = {}

    # ---- 1. LLM judge ------------------------------------------------------
    p = os.path.join(D, "1_stance_200.md")
    if os.path.exists(p):
        lab = parse(p, ["ABS"])
        pairs, miss = [], 0
        for i, v in lab.items():
            h = YESNO.get(v["ABS"])
            if h is None:
                miss += 1; continue
            pairs.append((h, bool(items[i]["auto"]["epistemic_abstention"])))
        res["stance"] = binary_report(f"[1] LLM judge — epistemic abstention  (unfilled {miss})", pairs)

    # ---- 2. WildGuard --------------------------------------------------------
    p = os.path.join(D, "2_wildguard_200.md")
    if os.path.exists(p):
        lab = parse(p, ["HARM", "REFUSE"])
        for tag, key, title in (("HARM", "harmful", "[2] WildGuard — harmfulness"),
                                ("REFUSE", "refusal", "[3] WildGuard — refusal")):
            pairs, miss = [], 0
            for i, v in lab.items():
                h = YESNO.get(v[tag])
                if h is None:
                    miss += 1; continue
                pairs.append((h, bool(items[i]["auto"][key])))
            res[key] = binary_report(f"{title}  (unfilled {miss})", pairs)

    # ---- 3/4. validity rule --------------------------------------------------
    for fn, title, note in (("3_validity_flagged_100.md", "[4] validity rule - flagged stratum (precision)",
                             "fraction the rule called broken that humans also called broken"),
                            ("4_validity_valid_100.md", "[5] validity rule - passed stratum (miss-rate)",
                             "fraction the rule called valid that humans called broken")):
        p = os.path.join(D, fn)
        if not os.path.exists(p):
            continue
        lab = parse(p, ["BROKEN"])
        hs = [(i, YESNO.get(v["BROKEN"])) for i, v in lab.items()]
        miss = sum(1 for _, h in hs if h is None)
        hs = [(i, h) for i, h in hs if h is not None]
        n = len(hs); k = sum(1 for _, h in hs if h)
        flagged = all(items[i]["auto"]["rule_flagged"] for i, _ in hs)
        rate = (k / n) if n else float("nan")
        # precision on the flagged stratum, miss rate on the valid stratum
        if flagged:
            val, lo, hi = rate, *wilson(k, n)
            lbl = "precision (human also broken)"
        else:
            val, lo, hi = rate, *wilson(k, n)
            lbl = "miss-rate (human broken)"
        print(f"\n{title}  (unfilled {miss})")
        print(f"  {note}")
        print(f"  n={n}  {lbl} = {k}/{n} = {val:.4f}  [95% CI {lo:.3f}, {hi:.3f}]")
        print(f"  kappa=undefined (rule assigns the same label to every item in the stratum)")
        res[fn] = {"n": n, "k": k, "rate": round(val, 4), "ci": [round(lo, 4), round(hi, 4)],
                   "stratum": "flagged" if flagged else "valid"}

    # ---- 5. frame classifier -------------------------------------------------
    p = os.path.join(D, "5_frame_200.md")
    if os.path.exists(p):
        lab = parse(p, ["FRAME"])
        pairs, miss, bad = [], 0, collections.Counter()
        for i, v in lab.items():
            h = v["FRAME"]
            if h not in FRAMES:
                miss += 1
                if h: bad[h] += 1
                continue
            pairs.append((h, items[i]["auto"]["frame"]))
        n = len(pairs)
        print(f"\n[6] frame classifier - 3-class  (unfilled/miswritten {miss}{'  ' + str(dict(bad)) if bad else ''})")
        if n:
            acc = sum(h == m for h, m in pairs) / n
            lo, hi = wilson(sum(h == m for h, m in pairs), n)
            k = kappa_multi(pairs, FRAMES)
            print(f"  n={n}  accuracy={acc:.4f} [95% CI {lo:.3f}, {hi:.3f}]  kappa={k:.4f}")
            print(f"  {'':12}" + "".join(f"{'auto:'+c:>14}" for c in FRAMES))
            for hc in FRAMES:
                row = [sum(1 for h, m in pairs if h == hc and m == mc) for mc in FRAMES]
                print(f"  human:{hc:<7}" + "".join(f"{v:>14}" for v in row))
            for c in FRAMES:
                tp = sum(1 for h, m in pairs if h == c and m == c)
                pr = sum(1 for _, m in pairs if m == c); rc = sum(1 for h, _ in pairs if h == c)
                print(f"  {c:10} precision={tp}/{pr}" + (f"={tp/pr:.3f}" if pr else "=n/a")
                      + f"   recall={tp}/{rc}" + (f"={tp/rc:.3f}" if rc else "=n/a"))
            res["frame"] = {"n": n, "accuracy": round(acc, 4), "kappa": round(k, 4)}

    out = os.path.join(D, "SCORES_200.json")
    json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
