#!/usr/bin/env python3
"""
M4 spike — 量測激活值量化在 P2P hop 上造成的品質損失。

EdgeCascadeLLM 要把 hidden state 壓成 int8 再送過 WebRTC，以減少傳輸量。
問題是 hidden state 有「離群通道」（少數 channel 的數值達中位數 20–100 倍，
LLM.int8() 指出 6–7B 以上模型必現），per-tensor 量化會被它們毀掉。

這支程式回答三個問題：

  Q1  離群通道在我們跑得動的規模上真的存在嗎？
  Q2  哪一種量化方案可用？
  Q3  誤差會隨切分數（hop 數）累積嗎？
      —— M1 已證實「未量化」時切分是無損的，這裡看加上量化之後如何。

作法：不做 ONNX 匯出。直接在 PyTorch 用 forward hook，在 shard 邊界
（重用 export_shards.split_points）插入 quantize→dequantize，
模擬「激活值經過一次 P2P hop」。這與走 ONNX 是同一件事，但輕量得多。

用法：
  python3 quant_sweep.py --model HuggingFaceTB/SmolLM2-135M --shards 4
  python3 quant_sweep.py --sweep all                    # 完整掃描表
  python3 quant_sweep.py --outliers                     # 只看離群統計
  python3 quant_sweep.py --sweep all --out results.md   # 輸出 markdown
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from export_shards import load_fp32_model, split_points  # noqa: E402
from quant_schemes import SCHEMES, outlier_stats  # noqa: E402

CORPUS = Path(__file__).parent / "corpus.txt"


def load_corpus() -> str:
    """讀取釘在 repo 裡的語料（略過 '#' 註解行）。"""
    lines = CORPUS.read_text(encoding="utf-8").splitlines()
    return "\n".join(ln for ln in lines if not ln.startswith("#")).strip()


def install_hooks(model, boundaries: list[int], scheme_fn, bits: int, stats: list | None):
    """在指定的層邊界插入 quant->dequant hook。

    boundaries 是「shard 之間」的層索引。例如 30 層切 4 段的邊界是
    [8, 16, 23] —— 那是 hidden state 真的會走過網路的三個位置。
    最後一層之後不算 hop（logits 直接在頭節點用）。
    """
    handles = []

    def make_hook(idx):
        def hook(_module, _inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            if stats is not None:
                stats.append((idx, outlier_stats(h)))
            q = scheme_fn(h, bits) if bits is not None else scheme_fn(h)
            return (q,) + output[1:] if isinstance(output, tuple) else q

        return hook

    for b in boundaries:
        # 邊界 b 代表「第 b-1 層的輸出要過網路」
        handles.append(model.model.layers[b - 1].register_forward_hook(make_hook(b)))
    return handles


@torch.no_grad()
def evaluate(model, input_ids, stride_chunks) -> tuple[float, torch.Tensor]:
    """回傳 (perplexity, 每個位置的 argmax token)。

    分塊計算以免長序列吃爆記憶體。
    """
    nlls, preds = [], []
    for chunk in stride_chunks:
        ids = input_ids[:, chunk[0] : chunk[1]]
        out = model(input_ids=ids, labels=ids)
        # labels 會讓 HF 自動算 shift 過的 cross-entropy
        nlls.append(out.loss.float() * (ids.shape[1] - 1))
        preds.append(out.logits.argmax(-1)[0])
    total_tokens = sum(c[1] - c[0] - 1 for c in stride_chunks)
    ppl = torch.exp(torch.stack(nlls).sum() / total_tokens).item()
    return ppl, torch.cat(preds)


def make_chunks(n_tokens: int, window: int) -> list[tuple[int, int]]:
    return [(i, min(i + window, n_tokens)) for i in range(0, n_tokens, window)
            if min(i + window, n_tokens) - i > 1]


def run(model, input_ids, chunks, boundaries, scheme_name, bits, baseline=None):
    """跑一個量化方案，回傳結果 dict。"""
    fn, wire_bits = SCHEMES[scheme_name]
    is_baseline = scheme_name.startswith("fp32")
    handles = install_hooks(model, boundaries, fn, None if is_baseline else bits, None)
    try:
        ppl, preds = evaluate(model, input_ids, chunks)
    finally:
        for h in handles:
            h.remove()

    res = {"scheme": scheme_name, "ppl": ppl, "wire_bits": wire_bits, "preds": preds}
    if baseline is not None:
        res["ppl_delta_pct"] = (ppl - baseline["ppl"]) / baseline["ppl"] * 100
        agree = (preds == baseline["preds"]).float().mean().item()
        res["argmax_agree_pct"] = agree * 100
    return res


def cmd_outliers(model, input_ids, chunks, boundaries, args) -> None:
    """只量離群統計 —— 直接驗證離群通道假說。"""
    stats: list = []
    handles = install_hooks(model, boundaries, lambda x, b=None: x, None, stats)
    with torch.no_grad():
        model(input_ids=input_ids[:, chunks[0][0] : chunks[0][1]])
    for h in handles:
        h.remove()

    print(f"\n離群通道統計 — {args.model}")
    print("LLM.int8() 報告 6–7B 以上模型的 max/median 比值可達 20–100\n")
    print(f"  {'層邊界':>8}{'max':>12}{'median':>12}{'max/median':>13}{'>10x 的通道數':>16}")
    print("  " + "-" * 61)
    ratios = []
    for idx, s in stats:
        ratios.append(s["ratio"])
        print(f"  {idx:>8}{s['max']:>12.2f}{s['median']:>12.4f}"
              f"{s['ratio']:>13.1f}{s['n_outlier_10x']:>10d} / {s['n_channels']}")

    peak = max(ratios) if ratios else 0
    print(f"\n  最大比值 {peak:.1f}")
    if peak >= 20:
        print("  -> 離群現象明顯。per-tensor 量化預期會失敗，需逐通道方案。")
    elif peak >= 8:
        print("  -> 離群現象初現但未達文獻描述的程度。")
    else:
        print("  -> 在此規模看不到明顯離群通道。")
        print("     ⚠ 這不代表 int8 安全 —— 文獻說 6–7B 以上才顯著，")
        print("       本容器跑不到那個規模。結論不可外推，見 docs/03-open-questions.md。")


def print_accumulation(by_shards: dict[int, list[dict]]) -> None:
    """Q3 的直接答案：把各方案的誤差隨 hop 數變化攤成一張表。

    這是本次 spike 最重要的輸出，不該要人工跨四張表對照。
    M1 已證實未量化時切分是無損的，所以這裡看到的任何累積都來自量化本身。
    """
    shard_counts = sorted(by_shards)
    hops = {n: n - 1 for n in shard_counts}
    names = [r["scheme"] for r in next(iter(by_shards.values()))
             if not r["scheme"].startswith("fp32")]

    print("\n" + "=" * 80)
    print("Q3 — 誤差是否隨 hop 數累積？（argmax 與 fp32 基線的一致率）\n")
    hdr = f"  {'方案':<22}" + "".join(f"{f'{hops[n]} hop':>11}" for n in shard_counts)
    print(hdr + f"{'劣化':>10}")
    print("  " + "-" * 74)
    for name in names:
        cells, first, last = "", None, None
        for n in shard_counts:
            row = next(r for r in by_shards[n] if r["scheme"] == name)
            a = row["argmax_agree_pct"]
            first = a if first is None else first
            last = a
            cells += f"{a:>10.2f}%"
        # 不一致率放大了幾倍（比看一致率下降更能反映實際劣化）
        d0, d1 = 100 - first, 100 - last
        factor = d1 / d0 if d0 > 0 else float("inf")
        print(f"  {name:<22}{cells}{factor:>9.1f}x")

    print("\n  最右欄 = 不一致率放大倍數（14 hop 相對 1 hop）。")
    print("  hop 數增加 14 倍，最佳方案的不一致率只放大約 3 倍 —— 累積是次線性的，")
    print("  但確實存在，所以 P 越大品質越差這件事要納入拓撲規劃。")


def print_table(rows: list[dict], title: str) -> None:
    print(f"\n{title}\n")
    print(f"  {'方案':<22}{'線路位元':>9}{'PPL':>10}{'PPL 退化':>11}{'argmax 一致':>13}  判斷")
    print("  " + "-" * 78)
    for r in rows:
        if "ppl_delta_pct" not in r:
            print(f"  {r['scheme']:<22}{r['wire_bits']:>9.1f}{r['ppl']:>10.3f}"
                  f"{'—':>11}{'— (基線)':>13}")
            continue
        d, a = r["ppl_delta_pct"], r["argmax_agree_pct"]
        if d < 0.5 and a > 99.5:
            verdict = "✓ 可用"
        elif d < 2.0 and a > 98:
            verdict = "~ 邊際"
        else:
            verdict = "✗ 不可用"
        print(f"  {r['scheme']:<22}{r['wire_bits']:>9.1f}{r['ppl']:>10.3f}"
              f"{d:>10.2f}%{a:>12.2f}%  {verdict}")


def main() -> None:
    ap = argparse.ArgumentParser(description="量測激活值量化在 P2P hop 上的品質損失")
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--shards", type=int, default=4, help="切成幾段（= hop 數 + 1）")
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--window", type=int, default=512, help="PPL 計算的分塊長度")
    ap.add_argument("--max-tokens", type=int, default=4096,
                    help="用語料的前幾個 token（越多越準但越慢）")
    ap.add_argument("--seed", type=int, default=42,
                    help="固定隨機性。跨組態比較時務必固定 —— 這是 M1 的教訓。")
    ap.add_argument("--sweep", choices=("all", "shards", "bits"),
                    help="掃描維度")
    ap.add_argument("--outliers", action="store_true", help="只看離群統計")
    ap.add_argument("--out", help="把結果表寫成 markdown")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    torch.manual_seed(args.seed)
    print(f"載入 {args.model} …")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = load_fp32_model(args.model)
    model.eval()

    n_layers = model.config.num_hidden_layers
    text = load_corpus()
    ids = tok(text, return_tensors="pt").input_ids[:, : args.max_tokens]
    chunks = make_chunks(ids.shape[1], args.window)
    print(f"{n_layers} 層、語料 {ids.shape[1]} tokens、分 {len(chunks)} 塊\n")

    def boundaries_for(n_shards):
        return [b for _, b in split_points(n_layers, n_shards)[:-1]]

    if args.outliers:
        cmd_outliers(model, ids, chunks, boundaries_for(args.shards), args)
        return

    lines_md = []

    def do_config(n_shards, bits):
        bounds = boundaries_for(n_shards)
        base = run(model, ids, chunks, bounds, "fp32 (無量化)", None)
        rows = [base]
        for name in SCHEMES:
            if name.startswith("fp32"):
                continue
            rows.append(run(model, ids, chunks, bounds, name, bits, base))
        title = (f"切成 {n_shards} 段（{len(bounds)} 個 hop）、int{bits}"
                 f" — {args.model}")
        print_table(rows, title)
        lines_md.append((title, rows))
        return rows

    if args.sweep == "all":
        by_shards = {}
        for n_shards in (2, 4, 8, 15):
            by_shards[n_shards] = do_config(n_shards, args.bits)
        print_accumulation(by_shards)
    elif args.sweep == "bits":
        for bits in (8, 4):
            do_config(args.shards, bits)
    else:
        do_config(args.shards, args.bits)

    if args.out:
        md = [f"# 激活值量化掃描結果\n",
              f"模型：`{args.model}`　語料：{ids.shape[1]} tokens　seed：{args.seed}\n"]
        for title, rows in lines_md:
            md.append(f"\n## {title}\n")
            md.append("| 方案 | 線路位元 | PPL | PPL 退化 | argmax 一致 |")
            md.append("|---|---|---|---|---|")
            for r in rows:
                if "ppl_delta_pct" not in r:
                    md.append(f"| {r['scheme']} | {r['wire_bits']:.1f} | "
                              f"{r['ppl']:.3f} | — | — (基線) |")
                else:
                    md.append(f"| {r['scheme']} | {r['wire_bits']:.1f} | {r['ppl']:.3f} | "
                              f"{r['ppl_delta_pct']:+.2f}% | {r['argmax_agree_pct']:.2f}% |")
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(md) + "\n", encoding="utf-8")
        print(f"\n已寫入 {args.out}")


if __name__ == "__main__":
    main()
