"""
激活值量化方案。

這些是 EdgeCascadeLLM 在 P2P hop 上壓縮 hidden state 的候選方案。
每個方案都是 quantize -> dequantize 的往返（我們不是要存 int8，
而是要模擬「傳輸時壓成 int8，收到後解回浮點」造成的資訊損失）。

這裡和 `web/src/quant.js` 是**同一組方案的兩份實作**，但**不是逐位元相同**，
不要寫任何 assert 兩邊相等的測試 —— 那個測試一定會紅，而且紅的原因不在測試裡。
（`quant.js` 的檔頭原本就寫著「必須逐位元對應」，那句話是假的，已經改掉。）

三個確定對不上的地方：

1. **捨入方向**：`torch.round` 是 half-to-even，JS 的 `Math.round` 是 half-up。
   `torch.round([2.5, -2.5, 3.5, 0.5])` = `[2, -2, 4, 0]`，
   同樣的輸入 JS 給的是 `[3, -2, 4, 1]`。這條吃到每一個被量化的值。
2. **fp16 往返**：這裡用 `.to(torch.float16)`（IEEE round-to-nearest-even，
   保留非正規數）；`quant.js` 的 `fp16Roundtrip()` 是手寫的位元操作，
   捨入是 half-up、非正規數直接沖成 ±0。
3. **離群比例的預設值**：這裡 `outlier_frac=0.01`，`quant.js` 是 `0.03`。
   連挑出來的離群 channel 集合都不一樣，所以 frame header 必須明確帶
   `outlierCount`（`docs/01-architecture.md` §4.5.2），不能靠約定反推。

兩邊該成立的是**統計上等價**（同方案、同語料下 PPL 與 argmax 一致率導向同一個結論），
不是位元相等。位元相等那個要求屬於 `web/src/wire.js` 的 encode/decode 往返。

另外：這個檔案和 `quant.js` 一樣都**不產生位元組** —— 每個函式回傳的是
quantize->dequantize 之後的張量。真正的線路序列化在 `web/src/wire.js`，
真實位元組帳在 `bench/wire_size.py`。

背景：hidden state 有「離群通道」—— 少數幾個 channel 的數值可達中位數的
20–100 倍，且在 6–7B 以上的模型必然出現（LLM.int8(), arXiv 2208.07339）。
per-tensor 量化的 scale 被離群值撐大之後，其餘 99.9% 的值只剩 2–3 個有效位元。
這就是為什麼這個模組要提供多種方案並實測比較，而不是直接選一個。
"""

from __future__ import annotations

import torch


def _qdq(x: torch.Tensor, scale: torch.Tensor, bits: int) -> torch.Tensor:
    """對稱量化的 quantize->dequantize 往返。

    scale 必須能廣播到 x 的形狀。回傳與 x 同形狀、同 dtype 的張量。
    """
    qmax = 2 ** (bits - 1) - 1
    # scale 為 0 代表該切片全是 0；避免除以 0，改用 1（量化結果仍是 0）
    safe = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.clamp(torch.round(x / safe), -qmax - 1, qmax)
    return q * safe


def per_tensor(x: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """整個張量共用一個 scale。

    最簡單、傳輸額外負擔最小（只要帶 1 個 float），
    但也最容易被離群通道毀掉 —— 這是要被驗證是否不可用的對照組。
    """
    qmax = 2 ** (bits - 1) - 1
    scale = x.abs().max() / qmax
    return _qdq(x, scale, bits)


def per_token(x: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """每個序列位置（token）一個 scale。

    離群通道是「跨 token 一致地出現在特定 channel」，所以 per-token
    其實幫助有限 —— 每個 token 的 scale 仍會被它自己的離群 channel 撐大。
    額外負擔：每個 token 一個 float。
    """
    qmax = 2 ** (bits - 1) - 1
    scale = x.abs().amax(dim=-1, keepdim=True) / qmax
    return _qdq(x, scale, bits)


def per_channel(x: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """每個 hidden 維度（channel）一個 scale。

    這是對付離群通道的正解：離群 channel 拿到自己的大 scale，
    其餘 channel 不受影響。
    額外負擔：每個 channel 一個 float，即 d_model 個 —— 注意這與
    K（序列長度）無關，所以 K 越大這個成本攤得越薄。
    """
    qmax = 2 ** (bits - 1) - 1
    dims = tuple(range(x.dim() - 1))  # 對 channel 以外的所有軸取 max
    scale = x.abs().amax(dim=dims, keepdim=True) / qmax
    return _qdq(x, scale, bits)


def group_wise(x: torch.Tensor, bits: int = 8, group: int = 128) -> torch.Tensor:
    """把 channel 切成固定大小的組，每組一個 scale。

    介於 per-tensor 與 per-channel 之間的折衷：離群值的影響被限制在
    它所屬的那一組內。額外負擔：d_model/group 個 float。
    """
    *lead, d = x.shape
    qmax = 2 ** (bits - 1) - 1
    n_full = (d // group) * group

    out = torch.empty_like(x)
    if n_full:
        xg = x[..., :n_full].reshape(*lead, n_full // group, group)
        scale = xg.abs().amax(dim=-1, keepdim=True) / qmax
        out[..., :n_full] = _qdq(xg, scale, bits).reshape(*lead, n_full)
    if n_full < d:
        # d_model 不是 group 的整數倍時，尾巴自成一組。
        # （例如 d_model=576 配 group=128 會剩 64 個 channel）
        tail = x[..., n_full:]
        scale = tail.abs().amax(dim=-1, keepdim=True) / qmax
        out[..., n_full:] = _qdq(tail, scale, bits)
    return out


def per_channel_outlier_fp16(
    x: torch.Tensor, bits: int = 8, outlier_frac: float = 0.01
) -> torch.Tensor:
    """LLM.int8() 式混合精度：離群 channel 保留高精度，其餘量化。

    先找出數值最大的一小撮 channel（預設 1%），那些 channel 原封不動
    （模擬用 fp16 傳輸），其餘走 per-channel 量化。

    傳輸成本：outlier_frac 的 channel 要用 2 bytes 而非 1 byte，
    所以 1% 離群時的額外負擔約 1%，很划算 —— 前提是它真的有效。
    """
    dims = tuple(range(x.dim() - 1))
    channel_mag = x.abs().amax(dim=dims)  # 每個 channel 的最大絕對值
    d = channel_mag.numel()
    n_out = max(1, int(d * outlier_frac))

    out = per_channel(x, bits)
    idx = torch.topk(channel_mag, n_out).indices
    # 離群 channel 用 fp16 往返（而非完全不動），才反映真實傳輸精度
    out[..., idx] = x[..., idx].to(torch.float16).to(x.dtype)
    return out


def per_token_outlier_fp16(
    x: torch.Tensor, bits: int = 8, outlier_frac: float = 0.01
) -> torch.Tensor:
    """per-token 量化 + 把數值最大的少數 channel 保留 fp16。

    與 per_channel_outlier_fp16 的差別在基底方案。實測發現離群同時具有
    token 集中性（attention sink）與 channel 集中性，所以兩種基底都要試。
    """
    dims = tuple(range(x.dim() - 1))
    channel_mag = x.abs().amax(dim=dims)
    n_out = max(1, int(channel_mag.numel() * outlier_frac))

    out = per_token(x, bits)
    idx = torch.topk(channel_mag, n_out).indices
    out[..., idx] = x[..., idx].to(torch.float16).to(x.dtype)
    return out


# 名稱 -> (函式, 每 token 每 channel 的等效傳輸位元數)
# 等效位元數用來公平比較：方案再準，如果傳輸量沒省就沒意義。
def _wire_bits_outlier(frac: float) -> float:
    """離群 channel 用 16 bits、其餘 8 bits 的等效平均位元數。"""
    return 8.0 * (1 - frac) + 16.0 * frac


SCHEMES = {
    "fp32 (無量化)": (lambda x, bits=None: x, 32.0),
    "per-tensor": (per_tensor, 8.0),
    "per-token": (per_token, 8.0),
    "per-channel": (per_channel, 8.0),
    "group-64": (lambda x, bits=8: group_wise(x, bits, 64), 8.0),
    "group-128": (lambda x, bits=8: group_wise(x, bits, 128), 8.0),
    "per-ch+outlier 0.5%": (
        lambda x, bits=8: per_channel_outlier_fp16(x, bits, 0.005),
        _wire_bits_outlier(0.005)),
    "per-ch+outlier 1%": (
        lambda x, bits=8: per_channel_outlier_fp16(x, bits, 0.01),
        _wire_bits_outlier(0.01)),
    "per-ch+outlier 3%": (
        lambda x, bits=8: per_channel_outlier_fp16(x, bits, 0.03),
        _wire_bits_outlier(0.03)),
    "per-tok+outlier 1%": (
        lambda x, bits=8: per_token_outlier_fp16(x, bits, 0.01),
        _wire_bits_outlier(0.01)),
}


def outlier_stats(x: torch.Tensor) -> dict:
    """量測離群程度：這是直接驗證「離群通道假說」的指標。

    回傳 max/median 比值 —— LLM.int8() 報告 6–7B 以上的模型會達到 20–100。
    若在小模型上這個比值只有個位數，代表在該規模上看不到這個現象，
    量化結論就不能外推到大模型。
    """
    dims = tuple(range(x.dim() - 1))
    mag = x.abs().amax(dim=dims).float()
    median = mag.median()
    return {
        "max": mag.max().item(),
        "median": median.item(),
        "ratio": (mag.max() / median).item() if median > 0 else float("inf"),
        # 超過中位數 10 倍的 channel 有幾個
        "n_outlier_10x": int((mag > median * 10).sum().item()),
        "n_channels": mag.numel(),
    }
