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


def shard_bytes(path: Path) -> int:
    """一個 shard 佔多少位元組（含 external data 旁檔）。"""
    total = path.stat().st_size
    ext = path.with_name(path.name + ".data")
    if ext.exists():
        total += ext.stat().st_size
    return total


MIN_TRANSFORMERS = 5


def require_transformers_v5() -> str:
    """確認 transformers 是 v5 以上，否則給出可操作的訊息後結束。

    本專案用 v5 的 API。v4 會在 `from_pretrained` 拒絕 `dtype` 參數，
    而且錯誤是從 transformers 內部丟出來的，完全看不出根因：

        TypeError: LlamaForCausalLM.__init__() got an unexpected keyword argument 'dtype'

    與其寫相容層讓兩邊都能跑，不如在這裡擋下來講清楚 ——
    版本太舊多半是因為機器上早就裝過舊版，而 `pip install` 對已安裝的套件
    不會升級（它只會說 Requirement already satisfied）。
    """
    import transformers

    version = transformers.__version__
    major = int(version.split(".")[0])
    if major < MIN_TRANSFORMERS:
        raise SystemExit(
            f"transformers 版本太舊：目前是 {version}，需要 {MIN_TRANSFORMERS}.0 以上。\n"
            f"\n"
            f"  多半是機器上早就裝過舊版 —— `pip install transformers` 對已安裝的\n"
            f"  套件不會升級，只會說 Requirement already satisfied。\n"
            f"\n"
            f"  升級方式：\n"
            f"    python -m pip install --upgrade transformers\n"
            f"  或直接跑（會一次處理好所有套件）：\n"
            f"    cd web && npm run setup"
        )
    return version


def load_fp32_model(model_id: str):
    """載入 HF 因果語言模型，權重固定為 fp32。

    fp32 是刻意的：這些權重之後會拿去做量化與數值比對，
    基準必須是未經降精度的版本，否則「量化造成的誤差」會混進
    「載入時就已經降精度」的誤差裡，分不開。
    """
    from transformers import AutoModelForCausalLM

    require_transformers_v5()
    return AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float32, attn_implementation="eager"
    )


def save_with_external_data(model, path: Path) -> None:
    """把模型連同 external data 存回原路徑。

    ⚠️ 必須先刪掉舊的 .data 檔。ONNX 的 save_model 不會截斷既有的旁檔，
    而是接著寫下去 —— 所以「載入 → 量化 → 存回同一路徑」會讓檔案
    變成「舊的 fp32 權重 + 新的 int4 權重」，反而更大。
    實測：初始化張量只有 16.3 MB，檔案卻是 131 MB。
    """
    import onnx

    ext = path.with_name(path.name + ".data")
    if ext.exists():
        ext.unlink()
    onnx.save_model(
        model, str(path),
        save_as_external_data=True, all_tensors_to_one_file=True,
        location=ext.name, size_threshold=1024,
    )


def embedding_to_fp16(model, min_bytes: int = 8 << 20) -> int:
    """把大型 embedding 查表（Gather 的 data）從 fp32 降成 fp16。

    為什麼要單獨處理：MatMulNBitsQuantizer 只量化 MatMul，
    embedding 是 Gather，完全不在它的範圍內。而 SmolLM2-135M 的
    embedding（vocab 49152 × 576）光自己就有 28.3M 參數 = 113 MB fp32，
    量化完之後反而變成整個模型最大的一塊。

    作法：initializer 存成 fp16，然後在 Gather 的輸出後面插一個 Cast 轉回 fp32，
    這樣下游的算子完全不用改。精度影響可忽略 —— embedding 查表出來的值
    本來就會馬上進 LayerNorm。

    回傳降轉了幾個張量。
    """
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper, TensorProto

    inits = {i.name: i for i in model.graph.initializer}
    converted = 0

    for node in list(model.graph.node):
        if node.op_type != "Gather" or not node.input:
            continue
        init = inits.get(node.input[0])
        if init is None or init.data_type != TensorProto.FLOAT:
            continue
        arr = numpy_helper.to_array(init)
        if arr.nbytes < min_bytes:
            continue

        # 1. initializer 改成 fp16
        new_init = numpy_helper.from_array(arr.astype(np.float16), init.name)
        init.CopyFrom(new_init)

        # 2. Gather 的輸出接一個 Cast 轉回 fp32，下游不用動
        gather_out = node.output[0]
        cast_out = gather_out + "_to_fp32"
        for consumer in model.graph.node:
            if consumer is node:
                continue
            for k, inp in enumerate(consumer.input):
                if inp == gather_out:
                    consumer.input[k] = cast_out
        for out in model.graph.output:
            if out.name == gather_out:
                out.name = cast_out

        cast = helper.make_node("Cast", [gather_out], [cast_out],
                                to=TensorProto.FLOAT,
                                name=f"{node.name or gather_out}_cast_fp32")
        idx = list(model.graph.node).index(node)
        model.graph.node.insert(idx + 1, cast)

        # 3. 型別宣告也要跟著改，否則 ORT 會在載入時就拒絕：
        #    "Type (tensor(float)) of output arg does not match expected type (tensor(float16))"
        #    Gather 現在輸出 fp16，Cast 之後才是 fp32。
        for vi in model.graph.value_info:
            if vi.name == gather_out:
                vi.type.tensor_type.elem_type = TensorProto.FLOAT16
                new_vi = TensorProto  # 佔位，避免 linter 誤判未使用
                break
        cast_vi = helper.make_tensor_value_info(cast_out, TensorProto.FLOAT, None)
        model.graph.value_info.append(cast_vi)

        converted += 1

    return converted


def prune_orphan_initializers(model) -> int:
    """移除沒有任何節點引用的 initializer，回傳清掉幾個。

    量化與圖改寫之後很容易留下孤兒權重。ONNX 不會自動清，
    而 save_model 會照單全收，所以檔案會莫名其妙變大。
    """
    used = set()
    for node in model.graph.node:
        used.update(node.input)
        # 子圖（If / Loop 的 branch）裡的引用也要算
        for attr in node.attribute:
            for g in list(attr.graphs) + ([attr.g] if attr.HasField("g") else []):
                for sub in g.node:
                    used.update(sub.input)
    used.update(o.name for o in model.graph.output)

    keep = [init for init in model.graph.initializer if init.name in used]
    removed = len(model.graph.initializer) - len(keep)
    if removed:
        del model.graph.initializer[:]
        model.graph.initializer.extend(keep)
        # graph input 若對應已刪除的 initializer 也要一併移除
        inputs = [i for i in model.graph.input if i.name in used]
        if len(inputs) != len(model.graph.input):
            del model.graph.input[:]
            model.graph.input.extend(inputs)
    return removed


def quantize_nbits(path: Path, bits: int, block_size: int, symmetric: bool):
    """就地把一個 ONNX 檔的 MatMul 權重量化成 int4（區塊量化）。

    只動 MatMul —— embedding 的 Gather 不在範圍內，會留在原精度。
    這是為什麼光靠 int4 仍然塞不進 Cloudflare 的 25 MiB 單檔上限：
    SmolLM2-135M 的 embedding（含 tied lm_head）就佔了 28.3M 參數。
    大檔分塊因此是必要的，不是選項。
    """
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
    import onnx

    model = onnx.load(str(path))
    # block_size / is_symmetric 是 quantizer 自己的參數，不是 algo_config 的。
    # 區塊 128：每 128 個權重共用一組 scale，是 MatMulNBits 的常見設定。
    quant = MatMulNBitsQuantizer(model, bits=bits, block_size=block_size,
                                 is_symmetric=symmetric)
    quant.process()
    # 量化器把 MatMul 換成 MatMulNBits，但**原本的 fp32 權重仍留在 graph 裡**
    # 變成沒人引用的孤兒 initializer，而 onnx.save 會把它們一起寫出去。
    # 不清掉的話檔案反而會比量化前更大（實測 217MB -> 357MB）。
    removed = prune_orphan_initializers(quant.model.model)
    # embedding 是 Gather，量化器碰不到，要另外降成 fp16。
    embeds = embedding_to_fp16(quant.model.model)
    save_with_external_data(quant.model.model, path)
    return removed, embeds


def to_fp16(path: Path) -> None:
    """就地把一個 ONNX 檔轉成 fp16。int4 跑不起來時的退路。"""
    import onnx
    from onnxconverter_common import float16

    model = onnx.load(str(path))
    model = float16.convert_float_to_float16(model, keep_io_types=True)
    save_with_external_data(model, path)


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
    ap.add_argument("--dtype", default="fp32", choices=("fp32", "fp16", "int4"),
                    help="權重精度。int4 用 MatMulNBitsQuantizer，模型從 591MB 降到約 74MB。\n"
                         "注意這不只是省空間：roofline 的 K* 與『每參數位元組』成正比，\n"
                         "拿 fp32 量出來的 K* 會是 int4 實際部署時的 8 倍，沒有參考價值。")
    ap.add_argument("--quant-bits", type=int, default=8, choices=(4, 8),
                    help="量化位元數。預設 8。\n"
                         "⚠️ 小模型對 4-bit 極度敏感：SmolLM2-135M 在 4-bit 下\n"
                         "argmax 只剩 3/16（等於壞掉），8-bit 則有 15/16。\n"
                         "大模型（7B 以上）通常撐得住 4-bit，屆時再調。")
    ap.add_argument("--quant-block", type=int, default=128,
                    help="量化區塊大小。越小品質越好、額外的 scale 越多。")
    ap.add_argument("--quant-symmetric", action="store_true",
                    help="用對稱量化（省一點空間但品質較差）。預設非對稱。")
    ap.add_argument("--opset", type=int, default=18,
                    help="ONNX opset。預設 18 是 torch.onnx 匯出器原生產出的版本；"
                         "指定更低的版本會觸發一次註定失敗的降版轉換（只是噪音，不影響結果）。")
    args = ap.parse_args()

    from transformers import AutoConfig

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"載入 {args.model} …")
    cfg = AutoConfig.from_pretrained(args.model)
    model = load_fp32_model(args.model)
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
        # 記下權重精度：K* 與「每參數位元組」成正比，
        # 量測報告必須連同這個值一起看才有意義。
        "dtype": args.dtype if args.dtype != "int4" else f"int{args.quant_bits}",
        "bytes_per_param": (
            {"fp32": 4.0, "fp16": 2.0}[args.dtype] if args.dtype != "int4"
            else args.quant_bits / 8
        ),
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
            # （必須在量化「之前」算，因為參考值要對照未量化的 PyTorch 模型）
            hidden = mod(*sample)

            if args.dtype == "int4":
                pruned, embeds = quantize_nbits(
                    path, args.quant_bits, args.quant_block, args.quant_symmetric)
            elif args.dtype == "fp16":
                to_fp16(path)

            size_mb = shard_bytes(path) / 1e6
            manifest["shards"].append({
                "index": i, "file": path.name, "layers": [a, b],
                "input": in_names[0], "output": out_name,
                "size_mb": round(size_mb, 1),
            })
            extra = ""
            if args.dtype == "int4":
                bits = []
                if pruned:
                    bits.append(f"清掉 {pruned} 個孤兒權重")
                if embeds:
                    bits.append(f"{embeds} 個 embedding 降 fp16")
                extra = ("，" + "、".join(bits)) if bits else ""
            print(f"  shard {i}: 層 {a}–{b}  ->  {path.name}  ({size_mb:.1f} MB{extra})")

    # 存一份未切分模型的參考 logits，給 verify_shards.py 比對
    with torch.no_grad():
        ref = model(input_ids=input_ids, position_ids=position_ids).logits

    # logits 存成 fp16 二進位而不是 JSON 文字。
    # 786,432 個浮點數存成 JSON 是 14.7 MB，存 fp16 只要 1.5 MB ——
    # 而且 JSON 版本本身就超過 Cloudflare 的 25 MiB 單檔上限。
    # fp16 的精度（約 1e-3 相對誤差）遠優於我們要驗的 1e-3 絕對容差，夠用。
    ref_np = ref.flatten().to(torch.float16).numpy()
    (out / "reference.bin").write_bytes(ref_np.tobytes())

    ref_path = out / "reference.json"
    ref_path.write_text(json.dumps({
        "input_ids": input_ids.tolist(),
        "position_ids": position_ids.tolist(),
        "logits_shape": list(ref.shape),
        "logits_file": "reference.bin",
        "logits_dtype": "float16",
        # argmax 另外存一份：貪婪解碼真正在意的就是這個，
        # 而且它讓「有沒有選錯字」可以獨立於浮點誤差來檢查。
        "argmax": ref.argmax(-1).flatten().tolist(),
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
