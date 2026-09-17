# 開發路線

每個里程碑都設計成**可單獨否證** —— 失敗時能明確知道是哪個假設錯了，
而不是「整體效果不好」。

| # | 里程碑 | 狀態 |
|---|---|---|
| M0 | 數值模型 | ✅ 完成 |
| M1 | 層切分 + hidden-state I/O（**go/no-go 閘門**） | ✅ **通過** |
| M2 | 瀏覽器 WebGPU 基線 + roofline 實測 | ⬜ 下一步 |
| M3 | 雙分頁 WebRTC 流水線 | ⬜ |
| M4 | 激活值量化 | ⬜ |
| M5 | 投機解碼 | ⬜ |
| M6 | Nostr 信令 + 真實 WAN | ⬜ |
| M7 | churn 容錯 | ⬜ |

---

## ✅ M0 — 數值模型

把延遲、roofline、churn 模型參數化，在寫任何推理程式碼之前先用第一性原理
估算瓶頸在哪。

**產出**：[`bench/model.py`](../bench/model.py)

**驗收**：`python3 bench/model.py --validate` 通過（對照 Petals 公開實測值，
預測值需落在 0.5×–2× 之間）。

**已達成**。過程中發現模型缺少「每 hop 固定開銷」項，補上後同時命中
Petals 的高延遲與低延遲測量點。這個發現改變了架構決策（P 要盡量少）。

---

## ✅ M1 — 層切分 + hidden-state I/O（go/no-go 閘門）

整個專案的前提：能把 Transformer 按層切開，讓 hidden state 在節點之間流動，
且結果與未切分時相同。

刻意**不碰瀏覽器、不碰網路、不碰量化、不碰 KV cache**，
把最根本的問題單獨隔離回答。

**產出**：[`spike/export_shards.py`](../spike/export_shards.py)、
[`spike/verify_shards.py`](../spike/verify_shards.py)

**驗收**：串接 N 個 shard 的 logits 與未切分模型 max abs diff < 1e-3，
且所有 argmax token 一致。

**已達成**：SmolLM2-135M（30 層）切 2/4/8/15 段全部通過，argmax 全數一致，
且固定輸入下誤差與切分數**完全無關**（一律 7.725e-05）——
切分本身是無損的。詳見 [00-feasibility.md §5](00-feasibility.md)。

```bash
pip install torch transformers onnx onnxruntime onnxscript
python3 spike/export_shards.py --model HuggingFaceTB/SmolLM2-135M --shards 4 --out out/ --seed 42
python3 spike/verify_shards.py --dir out/
```

---

## ⬜ M2 — 瀏覽器 WebGPU 基線 + roofline 實測

把 M1 的 shard 搬進瀏覽器，用 onnxruntime-web 的 WebGPU EP 跑起來。
量測不同 K 的實際吞吐，得出真實的 $K^*$。

**驗收**：
- 單一 shard 在瀏覽器中執行，輸出與 Node/Python 端一致
- 實測 $K^*$ 與模型預測相差 < 2×
- 量出真實的每 hop 固定開銷（含 WebGPU dispatch），回頭校正 `bench/model.py`

**否證條件**：若手機上 $K^* < 3$，位置平行的效益空間太小，
整個投機解碼路線要重新評估。

**已知風險**：ORT Web 的 WebGPU EP 對某些算子支援不全，可能需要改寫
匯出圖或 fallback 到 WASM。

---

## ⬜ M3 — 雙分頁 WebRTC 流水線

同一台機器上兩個分頁，走真實 WebRTC DataChannel 傳 hidden state。
先寫死信令（手動貼 SDP），不碰 Nostr。

**驗收**：
- 端到端輸出與單機執行一致
- 量測真實 per-hop 延遲與吞吐
- 正確處理 256 KiB 訊息上限（分片）與 `bufferedAmount` 背壓

**這一階段要加入 KV cache** —— M1 刻意跳過了它。KV cache 不影響正確性，
但不做的話每一步都要重算整個序列，效能無法看。

---

## ⬜ M4 — 激活值量化

每個 hop 把 fp16 激活值壓成 int8 以減少傳輸量。

**驗收**：
- 相對 fp16 基線的 PPL 退化 < 1%
- 傳輸量降到 1/2 以下
- **誤差 vs 切分數的曲線**。M1 已證實未量化時切分是無損的（誤差與切分數無關），
  所以這裡量到的任何累積都純粹來自 hop 量化 —— 這是一級指標，不是附註。
  **務必用 `--seed` 固定輸入**，否則量到的是輸入變異而不是組態差異

**必須用**逐通道/分組量化 + 離群通道 fp16。
per-tensor INT8 會因離群通道而毀掉品質，見
[01-architecture.md §4.2](01-architecture.md)。

---

## ⬜ M5 — 投機解碼

頭節點本地 draft model + 流水線一次驗證 γ+1 個 token。

**驗收**：
- 相對 M3 的 tok/s 提升 ≥ 2×
- **greedy 模式下輸出與自迴歸逐 token 相同**（這是硬性正確性條件）
- sampling 模式下通過分布等價性測試

---

## ⬜ M6 — Nostr 信令 + 真實 WAN

用 Nostr ephemeral events 交換 SDP，跨真實網路連線。

**驗收**：跨網路兩節點自動配對成功率、TURN 使用率有實際數據。

**必須包含 TURN fallback** —— NAT 直連會失敗一定比例。
這代表架構有一個中心化元件，文件要誠實寫出來。

---

## ⬜ M7 — churn 容錯

熱備節點 + activation checkpoint 續傳。

**驗收**：生成中途強制殺掉一個 stage 節點，對話不中斷。

---

## v1 範圍

> **私有信任 swarm + 32B 模型。**

不做開放公網、不做匿名節點、不做驗證機制、不做激勵機制。
Petals 證明了技術可行但網路效應才是殺手 —— 先讓「自己的三五台裝置
真的能跑 32B」這件事成立，再談規模。
