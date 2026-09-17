#!/usr/bin/env python3
"""
線路格式的真實位元組帳 —— 每個方案送一個 [K, d_model] 的 hidden state 要花多少。

## 為什麼需要這支程式

`docs/01-architecture.md` 原本寫「per-channel + 3% fp16 = 8.24 bits/值」，
而 `web/src/quant.js` 的 `SCHEMES.wire.wireBits` 也硬寫 8.24。

**那個數字只算了 int8 本體，沒有算必須跟著一起送的中繼資料。**

漏掉的兩項都不是可選的：

1. **scale 表**。`perChannel()`（quant.js:59-64）的 `scale[c]` 是
   `max|x[t][c]|` 對「這一則訊息裡的這 K 個 token」取的 —— 它跟著資料走，
   接收端推不出來、也不能快取。所以每則訊息都要帶 d 個（或 d-nOut 個）scale。

2. **離群 channel 的索引表**。`perChannelOutlierFp16()`（quant.js:105-131）
   每次都重新挑一次離群集合，所以每則訊息的集合都不一樣，也必須跟著送。

漏算的後果不是小數點問題：K=8 時真實成本是 10.24 bits/值，不是 8.24 —— 差 24%。
而且方案的排名會反轉：把中繼資料算進去之後，group-64 在任何實際會用到的 K
底下都比較省。這支程式就是為了讓那張表是「算出來的」而不是「手打的」。

## 帳怎麼算

`wireBytes()`（quant.js:156-159）做的是 `ceil(numel * wireBits / 8)` ——
對一個「數量」做算術，從來沒碰過資料。這裡改成描述真正的位元組佈局。

共通符號：d = d_model，K = 這則訊息裡的 token 位置數，S = scale 的位元組數。
"""

import argparse
import math

# scale 用 fp16 送。理由：scale 本身是用來把數值量化到 int8 的，
# fp16 的 2^-11 相對誤差遠小於它所乘上的 2^-8 量化步階。
# 但 fp32 也列出來，因為兩個方案的交叉點對這個選擇非常敏感。
SCALE_BYTES = {"fp16": 2, "fp32": 4}

OUTLIER_FRAC = 0.03  # quant.js:149 的預設值


def n_outliers(d, frac=OUTLIER_FRAC):
    """與 quant.js:117 完全一致：max(1, floor(d * frac))。"""
    return max(1, math.floor(d * frac))


def outlier_index_bytes(d, n_out):
    """
    離群索引的編碼方式取兩者中較省的：
      - u16 索引表：n_out * 2 位元組
      - bitmap：    ceil(d / 8) 位元組
    3% 的離群比例下索引表一定勝出（3% < 1/16 = 6.25%），但把兩者都算出來
    比較誠實 —— 如果之後把 outlier_frac 調高，交叉點會自己出現。
    """
    return min(n_out * 2, math.ceil(d / 8))


def layout(scheme, d, k, scale_prec="fp16"):
    """回傳這一則訊息的位元組拆帳（不含 frame header）。"""
    s = SCALE_BYTES[scale_prec]
    numel = d * k

    if scheme == "none":
        return {"body": 4 * numel, "scales": 0, "index": 0}

    if scheme == "per-channel":
        # d 個 scale，K 個 token 共用
        return {"body": numel, "scales": d * s, "index": 0}

    if scheme == "group-64":
        # 每個 token、每 64 個 channel 一個 scale。
        # d 不是 64 的整數倍時尾巴自成一組（quant.js:88-89）。
        groups = math.ceil(d / 64)
        return {"body": numel, "scales": k * groups * s, "index": 0}

    if scheme == "wire":
        # per-channel int8 + 前 3% 大的 channel 改送 fp16。
        # 離群 channel 不需要 scale（它們不走 int8 路徑），所以 scale 表是 d - n_out。
        n_out = n_outliers(d)
        return {
            "body": (d - n_out) * k + n_out * k * 2,
            "scales": (d - n_out) * s,
            "index": outlier_index_bytes(d, n_out),
        }

    raise ValueError(f"未知方案 {scheme}")


def total_bytes(scheme, d, k, scale_prec="fp16"):
    return sum(layout(scheme, d, k, scale_prec).values())


def bits_per_value(scheme, d, k, scale_prec="fp16"):
    return total_bytes(scheme, d, k, scale_prec) * 8 / (d * k)


def crossover_k(d, scale_prec="fp16", kmax=100_000):
    """wire 變得比 group-64 省的最小 K。找不到就回傳 None。"""
    for k in range(1, kmax + 1):
        if bits_per_value("wire", d, k, scale_prec) <= bits_per_value("group-64", d, k, scale_prec):
            return k
    return None


# M4 實測的品質數字，來自 docs/01-architecture.md §4.3 / docs/data/quant-results.md。
# 放在這裡是為了讓「省多少位元組」和「賠多少品質」出現在同一張表上 ——
# 只看其中一邊沒辦法做決定。
QUALITY = {
    "none": ("—", "100%"),
    "per-channel": ("+6.23%", "—"),
    "group-64": ("+0.48%", "97.56%"),
    "wire": ("+0.03%", "99.32%"),
}

SCHEMES = ["none", "per-channel", "group-64", "wire"]


def print_table(dims, ks, scale_prec):
    print(f"每個值的線路成本（bits/值，scale 用 {scale_prec}）")
    print()
    for d in dims:
        n_out = n_outliers(d)
        print(f"d_model = {d}（離群 channel {n_out} 個，group-64 有 {math.ceil(d / 64)} 組）")
        head = "  方案".ljust(16) + "".join(f"K={k}".rjust(9) for k in ks)
        head += "   PPL 退化   argmax@3hop"
        print(head)
        print("  " + "-" * (len(head) - 2))
        for s in SCHEMES:
            row = f"  {s}".ljust(16)
            row += "".join(f"{bits_per_value(s, d, k, scale_prec):9.2f}" for k in ks)
            ppl, argmax = QUALITY[s]
            row += f"{ppl:>11}{argmax:>14}"
            print(row)
        x = crossover_k(d, scale_prec)
        print(f"  -> wire 要 K ≥ {x} 才比 group-64 省" if x else "  -> wire 在任何 K 都不比 group-64 省")
        print()


def print_breakdown(d, k, scale_prec):
    print(f"位元組拆帳：d_model={d}, K={k}, scale={scale_prec}")
    print()
    print("  方案".ljust(16) + "本體".rjust(10) + "scale".rjust(10) + "索引".rjust(8)
          + "合計".rjust(10) + "bits/值".rjust(10))
    print("  " + "-" * 62)
    for s in SCHEMES:
        L = layout(s, d, k, scale_prec)
        t = sum(L.values())
        print(f"  {s}".ljust(16)
              + f"{L['body']:>10}" + f"{L['scales']:>10}" + f"{L['index']:>8}"
              + f"{t:>10}" + f"{bits_per_value(s, d, k, scale_prec):>10.2f}")
    print()
    print("  注意 scale 那一欄：K 小的時候它就是主導項，而 K 正是投機解碼的猜測長度。")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dims", type=int, nargs="+", default=[576, 5120],
                    help="要列出的 d_model（預設：576=SmolLM2-135M，5120=32B 級）")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 4, 8, 16, 32],
                    help="要列出的位置數 K")
    ap.add_argument("--scale", choices=list(SCALE_BYTES), default="fp16")
    ap.add_argument("--breakdown", action="store_true", help="印出單一組合的位元組拆帳")
    ap.add_argument("--both-precisions", action="store_true",
                    help="fp16 與 fp32 兩種 scale 都印（交叉點對這個很敏感）")
    args = ap.parse_args()

    if args.breakdown:
        for d in args.dims:
            print_breakdown(d, args.ks[len(args.ks) // 2], args.scale)
        return

    precisions = list(SCALE_BYTES) if args.both_precisions else [args.scale]
    for p in precisions:
        print_table(args.dims, args.ks, p)


if __name__ == "__main__":
    main()
