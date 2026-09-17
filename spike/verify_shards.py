#!/usr/bin/env python3
"""
M1 spike — 驗證切分後的 ONNX shard 串接起來與未切分模型數值等價。

這是 go/no-go 閘門的第二半。export_shards.py 已經在 PyTorch 層級確認過切分正確，
這支程式再確認一次「ONNX 匯出 + ONNX Runtime 執行」沒有把等價性弄丟。

為什麼要分兩步驗：切分邏輯錯（層次錯、mask 錯、RoPE 位置錯）和匯出錯
（算子語意漂移、常數摺疊失誤、精度降級）是兩種完全不同的 bug，
分開驗才知道要修哪裡。

同時這支程式會量測真實的 shard 間傳輸量 —— 那就是 WebRTC 上要傳的東西，
可以拿回去校正 bench/model.py 的估計。

用法：
  python3 verify_shards.py --dir out/
  python3 verify_shards.py --dir out/ --tolerance 1e-3
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def main() -> None:
    ap = argparse.ArgumentParser(description="驗證 ONNX shard 串接的數值等價性")
    ap.add_argument("--dir", default="out", help="export_shards.py 的輸出目錄")
    ap.add_argument("--tolerance", type=float, default=1e-3,
                    help="logits 的 max abs diff 容忍上限")
    args = ap.parse_args()

    d = Path(args.dir)
    manifest = json.loads((d / "manifest.json").read_text())
    ref_data = json.loads((d / "reference.json").read_text())

    input_ids = np.array(ref_data["input_ids"], dtype=np.int64)
    position_ids = np.array(ref_data["position_ids"], dtype=np.int64)
    ref_logits = np.array(ref_data["logits"], dtype=np.float32).reshape(
        ref_data["logits_shape"])

    print(f"模型      {manifest['model']}")
    print(f"切分      {manifest['num_layers']} 層 -> {manifest['num_shards']} 個 shard "
          f"{manifest['bounds']}")
    print(f"d_model   {manifest['hidden_size']}")
    print(f"序列長度  {input_ids.shape[1]}\n")

    # 依序執行每個 shard，把上一個的輸出餵給下一個 —— 這就是 P2P 流水線在做的事
    tensor = input_ids
    hop_bytes = []
    print(f"{'shard':>6}{'層':>10}{'耗時ms':>10}{'輸出形狀':>22}{'傳輸量':>12}")
    print("-" * 62)

    for meta in manifest["shards"]:
        path = d / meta["file"]
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

        feeds = {meta["input"]: tensor, "position_ids": position_ids}
        t0 = time.perf_counter()
        (out,) = sess.run([meta["output"]], feeds)
        dt = (time.perf_counter() - t0) * 1000

        is_last = meta["index"] == manifest["num_shards"] - 1
        if not is_last:
            # 這個張量就是要走 WebRTC 的 payload
            hop_bytes.append(out.nbytes)
            payload = f"{out.nbytes / 1024:.1f} KB"
        else:
            payload = "— (logits)"

        a, b = meta["layers"]
        print(f"{meta['index']:>6}{f'{a}–{b}':>10}{dt:>10.1f}"
              f"{str(out.shape):>22}{payload:>12}")
        tensor = out

    got = tensor.astype(np.float32)

    print("\n" + "=" * 62)
    if got.shape != ref_logits.shape:
        print(f"✗ 形狀不符：得到 {got.shape}，期望 {ref_logits.shape}")
        raise SystemExit(1)

    max_diff = float(np.abs(got - ref_logits).max())
    mean_diff = float(np.abs(got - ref_logits).mean())

    # 對貪婪解碼而言，真正要緊的是 argmax 有沒有變 —— logits 差一點沒關係，
    # 選出不同的 token 才是災難。
    got_tok = got.argmax(-1).ravel()
    ref_tok = ref_logits.argmax(-1).ravel()
    tok_match = int((got_tok == ref_tok).sum())
    tok_total = int(got_tok.size)

    print(f"logits max abs diff   {max_diff:.3e}   (容忍上限 {args.tolerance:.0e})")
    print(f"logits mean abs diff  {mean_diff:.3e}")
    print(f"argmax token 一致     {tok_match}/{tok_total}")

    if hop_bytes:
        seq = input_ids.shape[1]
        per_tok = sum(hop_bytes) / len(hop_bytes) / seq
        print(f"\n每個 hop 的傳輸量     {hop_bytes[0] / 1024:.1f} KB "
              f"(fp32, seq={seq})")
        print(f"換算每 token 每 hop   {per_tok:.0f} B (fp32) / "
              f"{per_tok / 4:.0f} B (int8) — 可拿去校正 bench/model.py")

    print("=" * 62)
    ok = max_diff < args.tolerance and tok_match == tok_total
    if ok:
        print("\n✓ M1 通過：ONNX shard 串接與未切分模型數值等價。")
        print("  EdgeCascadeLLM 的核心前提成立，可以往 M2（瀏覽器 + WebGPU）推進。")
    else:
        print("\n✗ M1 未通過。")
        if tok_match != tok_total:
            print("  argmax 不一致代表這不只是精度問題，是切分邏輯有錯。")
            print("  優先檢查：RoPE 的 position_ids 是否在每個 shard 都正確、")
            print("            causal mask 是否被重複套用、層的邊界有無 off-by-one。")
        else:
            print("  argmax 一致但 logits 偏差超標，可能是 ONNX 匯出的精度問題。")
            print("  檢查 do_constant_folding 與 opset 版本。")
        print("  把失敗原因寫進 docs/03-open-questions.md。")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
