"""分析候選文字長度分布, 評估不同 CHAR_LIMIT 的截斷比例"""
import os
import statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from dataset import load_train, load_test, build_candidates, split_train_dev


def collect_lengths(samples):
    text_lens, img_lens = [], []
    for s in samples:
        for c in build_candidates(s):
            L = len((c["text_for_retrieval"] or ""))
            (text_lens if c["modality"] == "text" else img_lens).append(L)
    return text_lens, img_lens


def trunc_pct(lens, limit):
    if not lens:
        return 0.0
    return 100.0 * sum(1 for L in lens if L > limit) / len(lens)


def stats_line(name, lens):
    if not lens:
        return f"  {name}: (empty)"
    s = sorted(lens)
    return (f"  {name}: n={len(lens)}  mean={st.mean(lens):.0f}  median={st.median(lens):.0f}  "
            f"p90={s[int(len(lens)*0.9)]}  p99={s[int(len(lens)*0.99)]}  max={max(lens)}")


def main():
    train = load_train()
    test  = load_test()
    _, dev = split_train_dev(train)

    splits = {
        "all (train+test)": train + test,
        "train (n=2055)": train,
        "test (n=1798)": test,
        "dev (n=200)":   dev,
    }

    LIMITS = [800, 1000, 1500, 2000]
    print("=" * 90)
    print("Candidate length statistics & truncation rate at CHAR_LIMIT")
    print("=" * 90)

    rows = []
    for name, data in splits.items():
        text_lens, img_lens = collect_lengths(data)
        print(f"\n[{name}]")
        print(stats_line("text  desc", text_lens))
        print(stats_line("image desc", img_lens))
        print(f"  {'limit':>6} | {'text>limit':>12} | {'image>limit':>12}")
        for lim in LIMITS:
            tp = trunc_pct(text_lens, lim)
            ip = trunc_pct(img_lens, lim)
            print(f"  {lim:>6} | {tp:>11.2f}% | {ip:>11.2f}%")
            rows.append((name, lim, tp, ip))

    # ── plot ──────────────────────────────────────────
    text_all, img_all = collect_lengths(train + test)
    out_dir = os.path.join(config.OUTPUT_DIR, "phase_6", "hparam")
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bins = list(range(0, 4001, 100))

    for ax, lens, title, color in [
        (axes[0], text_all, f"Text descriptions (n={len(text_all):,})", "#3b82f6"),
        (axes[1], img_all,  f"Image descriptions (n={len(img_all):,})", "#f59e0b"),
    ]:
        clipped = [min(L, 4000) for L in lens]
        ax.hist(clipped, bins=bins, color=color, edgecolor="white", alpha=0.85)
        for lim, ls in zip(LIMITS, ["--", "-.", ":", "-"]):
            pct = trunc_pct(lens, lim)
            ax.axvline(lim, color="red", linestyle=ls, linewidth=1.2,
                       label=f"limit={lim} (trunc {pct:.1f}%)")
        ax.set_title(title)
        ax.set_xlabel("length (chars, clipped at 4000 for display)")
        ax.set_ylabel("count")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle("INLP HW3 — Candidate text length distribution (train + test combined)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    out_png = os.path.join(out_dir, "char_length_distribution.png")
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    print(f"\n📊 figure saved → {out_png}")


if __name__ == "__main__":
    main()
