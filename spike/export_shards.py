#!/usr/bin/env python3
"""
M1 spike — 把一個 HF 因果語言模型垂直切成 N 個 ONNX shard。

這是 EdgeCascadeLLM 的 go/no-go 閘門。整個專案的前提是：
「能把 Transformer 按層切開，讓 hidden state 在節點之間流動，且結果與未切分時相同」。
這支程式驗證那個前提，而且刻意**不碰瀏覽器、不碰網路、不碰量化** ——
先把最根本的問題單獨隔離出來回答。

切分方式：
  shard 0      : embedding + layers[0:n]          -> hidden_states
  shard 1..N-2 : layers[a:b]  hidden_states       -> hidden_states
  shard N-1    : layers[a:] + final_norm + lm_head -> logits

每個中間 shard 的介面就是 EdgeCascadeLLM 在 WebRTC 上要傳的東西：
一個 [batch, seq, d_model] 的張量。

用法：
  python3 export_shards.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --shards 4
  python3 export_shards.py --model <hf-id> --shards 2 --out out/ --seq-len 32

注意：這一版**不處理 KV cache**（每個 shard 都吃完整序列重算）。
理由是先把「切分是否數值等價」這件事單獨驗證掉；KV cache 是 M3 的事，
且它只影響效能，不影響正確性。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn


class EmbedShard(nn.Module):
    """第一個 shard：token ids -> hidden states。"""

    def __init__(self, model, end: int):
        super().__init__()
        self.embed = model.model.embed_tokens
        self.layers = nn.ModuleList(model.model.layers[:end])
        self.rotary = getattr(model.model, "rotary_emb", None)

    def forward(self, input_ids, position_ids):
        h = self.embed(input_ids)
        pos = _rope(self.rotary, h, position_ids)
        mask = _causal_mask(h)
        for layer in self.layers:
            h = _run_layer(layer, h, mask, position_ids, pos)
        return h


class MiddleShard(nn.Module):
    """中間 shard：hidden states -> hidden states。這是 P2P 上流動的東西。"""

    def __init__(self, model, start: int, end: int):
        super().__init__()
        self.layers = nn.ModuleList(model.model.layers[start:end])
        self.rotary = getattr(model.model, "rotary_emb", None)

    def forward(self, hidden_states, position_ids):
        h = hidden_states
        pos = _rope(self.rotary, h, position_ids)
        mask = _causal_mask(h)
        for layer in self.layers:
            h = _run_layer(layer, h, mask, position_ids, pos)
        return h


class HeadShard(nn.Module):
    """最後一個 shard：hidden states -> logits。"""

    def __init__(self, model, start: int):
        super().__init__()
        self.layers = nn.ModuleList(model.model.layers[start:])
        self.norm = model.model.norm
        self.lm_head = model.lm_head
        self.rotary = getattr(model.model, "rotary_emb", None)

    def forward(self, hidden_states, position_ids):
        h = hidden_states
        pos = _rope(self.rotary, h, position_ids)
        mask = _causal_mask(h)
        for layer in self.layers:
            h = _run_layer(layer, h, mask, position_ids, pos)
        return self.lm_head(self.norm(h))


def _rope(rotary, hidden, position_ids):
    """較新的 transformers 把 RoPE 提到 model 層級統一算，舊版在每個 layer 內算。"""
    if rotary is None:
        return None
    return rotary(hidden, position_ids)


def _causal_mask(hidden):
    """因果遮罩。用加法式 mask（0 / -inf），對所有 attention 實作都通用。"""
    seq = hidden.shape[1]
    mask = torch.full((seq, seq), torch.finfo(hidden.dtype).min, dtype=hidden.dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask[None, None, :, :].expand(hidden.shape[0], 1, seq, seq)


def _run_layer(layer, hidden, mask, position_ids, pos_emb):
    """呼叫一個 decoder layer，吸收不同 transformers 版本的簽章差異。"""
    kwargs = {"attention_mask": mask, "position_ids": position_ids}
    if pos_emb is not None:
        kwargs["position_embeddings"] = pos_emb
    out = layer(hidden, **kwargs)
    return out[0] if isinstance(out, tuple) else out


def split_points(n_layers: int, n_shards: int) -> list[tuple[int, int]]:
    """把 n_layers 平均分成 n_shards 段，餘數分給前面幾段。"""
    if n_shards > n_layers:
        raise SystemExit(f"shard 數 {n_shards} 不能超過層數 {n_layers}")
    base, rem = divmod(n_layers, n_shards)
    bounds, cur = [], 0
    for i in range(n_shards):
        size = base + (1 if i < rem else 0)
        bounds.append((cur, cur + size))
        cur += size
    return bounds


def main() -> None:
    ap = argparse.ArgumentParser(description="把 HF 模型切成 N 個 ONNX shard")
    ap.add_argument("--model", required=True, help="HF model id 或本地路徑")
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--out", default="out", help="輸出目錄")
    ap.add_argument("--seq-len", type=int, default=16, help="匯出時的樣本序列長度")
    ap.add_argument("--seed", type=int, default=0,
                    help="隨機輸入的種子。比較不同組態（例如不同 shard 數）時務必固定，"
                         "否則量到的是輸入差異而不是組態差異。")
    ap.add_argument("--opset", type=int, default=18,
                    help="ONNX opset。預設 18 是 torch.onnx 匯出器原生產出的版本；"
                         "指定更低的版本會觸發一次註定失敗的降版轉換（只是噪音，不影響結果）。")
    args = ap.parse_args()

    from transformers import AutoConfig, AutoModelForCausalLM

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"載入 {args.model} …")
    cfg = AutoConfig.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()

    n_layers = cfg.num_hidden_layers
    d_model = cfg.hidden_size
    bounds = split_points(n_layers, args.shards)
    print(f"{n_layers} 層、d_model={d_model} -> 切成 {args.shards} 段：{bounds}\n")

    seq = args.seq_len
    torch.manual_seed(args.seed)
    input_ids = torch.randint(0, cfg.vocab_size, (1, seq), dtype=torch.int64)
    position_ids = torch.arange(seq, dtype=torch.int64)[None, :]
    dyn = {0: "batch", 1: "seq"}

    manifest = {
        "model": args.model,
        "num_layers": n_layers,
        "hidden_size": d_model,
        "vocab_size": cfg.vocab_size,
        "num_shards": args.shards,
        "bounds": bounds,
        "shards": [],
    }

    with torch.no_grad():
        hidden = None
        for i, (a, b) in enumerate(bounds):
            is_first, is_last = i == 0, i == args.shards - 1
            path = out / f"shard_{i}.onnx"

            if is_first:
                mod = EmbedShard(model, b)
                sample = (input_ids, position_ids)
                in_names = ["input_ids", "position_ids"]
                dynamic = {"input_ids": dyn, "position_ids": dyn}
            elif is_last:
                mod = HeadShard(model, a)
                sample = (hidden, position_ids)
                in_names = ["hidden_states", "position_ids"]
                dynamic = {"hidden_states": dyn, "position_ids": dyn}
            else:
                mod = MiddleShard(model, a, b)
                sample = (hidden, position_ids)
                in_names = ["hidden_states", "position_ids"]
                dynamic = {"hidden_states": dyn, "position_ids": dyn}

            out_name = "logits" if is_last else "hidden_states_out"
            dynamic[out_name] = dyn
            mod.eval()

            torch.onnx.export(
                mod, sample, str(path),
                input_names=in_names, output_names=[out_name],
                dynamic_axes=dynamic, opset_version=args.opset,
                do_constant_folding=True,
            )

            # 把這個 shard 的真實輸出接給下一個 shard，確保匯出時的樣本輸入是真的
            hidden = mod(*sample)

            size_mb = path.stat().st_size / 1e6
            manifest["shards"].append({
                "index": i, "file": path.name, "layers": [a, b],
                "input": in_names[0], "output": out_name,
                "size_mb": round(size_mb, 1),
            })
            print(f"  shard {i}: 層 {a}–{b}  ->  {path.name}  ({size_mb:.1f} MB)")

    # 存一份未切分模型的參考 logits，給 verify_shards.py 比對
    with torch.no_grad():
        ref = model(input_ids=input_ids, position_ids=position_ids).logits

    ref_path = out / "reference.json"
    ref_path.write_text(json.dumps({
        "input_ids": input_ids.tolist(),
        "position_ids": position_ids.tolist(),
        "logits_shape": list(ref.shape),
        "logits": ref.flatten().tolist(),
    }))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Python 端先自己比一次：串接 shard 的結果 vs 未切分模型
    diff = (hidden - ref).abs().max().item()
    print(f"\n參考 logits 形狀 {list(ref.shape)} -> {ref_path.name}")
    print(f"manifest -> manifest.json")
    print(f"\nPyTorch 端自檢：串接 shard 與未切分模型的 max abs diff = {diff:.3e}")
    if diff < 1e-3:
        print("✓ 切分在 PyTorch 層級數值等價。接著跑 verify_shards.py 驗證 ONNX 端。")
    else:
        print("✗ 切分在 PyTorch 層級就已經不等價了 —— ONNX 匯出無需再試，先修這裡。")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
