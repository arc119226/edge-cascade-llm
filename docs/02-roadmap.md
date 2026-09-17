# 開發路線

每個里程碑都設計成**可單獨否證** —— 失敗時能明確知道是哪個假設錯了，
而不是「整體效果不好」。

| # | 里程碑 | 狀態 |
|---|---|---|
| M0 | 數值模型 | ✅ 完成 |
| M1 | 層切分 + hidden-state I/O（**go/no-go 閘門**） | ✅ **通過** |
| M2 | 瀏覽器端執行 + 效能量測工具 | ✅ **完成**（WebGPU 數字待實測回報） |
| M3 | 雙分頁 WebRTC 流水線 | ⬜ 下一步 |
| M4 | 激活值量化 | ✅ **完成** |
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

## ✅ M2 — 瀏覽器端執行 + 效能量測工具

把 M1 的 shard 搬進瀏覽器，用 onnxruntime-web 跑起來。

**產出**：[`web/`](../web/) —— 可直接部署到 Cloudflare Pages 的靜態站，
含 shard 流水線執行器、激活值量化（與 Python 版逐值一致）、
權重快取（Cache API + service worker）、PWA、以及一鍵式效能量測頁。

**驗收（容器內可驗的部分）**：全數通過。
`npm test` 用 headless Chromium 跑完整條流水線，
4 段 / 30 層，max abs diff **1.502e-04**，argmax **16/16 一致**。

**刻意的分工**：規劃時實測確認**這個開發容器完全沒有 WebGPU**
（headless shell、完整 chromium、Xvfb + headed、swiftshader 全試過，
`navigator.gpu` 一律不存在）。所以 M2 拆成兩半：

| 半邊 | 誰做 | 狀態 |
|---|---|---|
| 正確性（WASM EP） | 容器內 `npm test` | ✅ 通過 |
| 效能與 WebGPU 正確性 | 真實裝置開 `bench.html` | ⬜ 待回報 |

`bench.html` 會量 $K^*$、每 hop 固定開銷、以及 WASM 與 WebGPU 的數值差異，
輸出一段 JSON 貼回來校正 `bench/model.py`。
在拿到真實數據之前，那些數字在文件中一律標為**未實測**。

**否證條件仍然有效**：若手機上 $K^* < 3$，位置平行的效益空間太小，
投機解碼路線要重新評估。

**過程中踩到並解決的三件事**：
1. `.onnx.data` 旁檔 ORT-Web 不會自己抓，必須用 `externalData` 明確餵進去
2. `wasmPaths` 是相對於 ORT 自己的模組網址解析，不是相對於頁面
3. WebGPU 版的 wasm 是 28 MB，**超過 Cloudflare Pages 的 25 MiB 單檔上限** ——
   建置改成把大檔分流到 R2，見 [DEPLOY.md](DEPLOY.md)

```bash
cd web && npm ci && npm run build && node scripts/prepare-model.mjs
npm test
```

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

## ✅ M4 — 激活值量化

每個 hop 把激活值壓成 int8 以減少傳輸量。
**提前到 M2 之前做**，因為它是 P0 風險且完全可在本地驗證 ——
若 int8 不可用，整個可行性評估要重算。

**產出**：[`spike/quant_sweep.py`](../spike/quant_sweep.py)、
[`spike/quant_schemes.py`](../spike/quant_schemes.py)、
[`spike/corpus.txt`](../spike/corpus.txt)（釘在 repo 的語料，不下載）

**驗收**：PPL 退化 < 1%、傳輸量減半、量出誤差 vs 切分數曲線。**全數達成。**

**結論**：採用 `per-channel + 前 3% 通道 fp16`（線路 8.24 bits）——
PPL 退化 **0.03%**、argmax 一致 **99.32%**。
純 int8 的最佳選項是 `group-64`（+0.48%，零額外頻寬）。

**三個實測發現**：
1. 離群比值在 **135M** 就達 **1212×**，遠超文獻對 6–7B 報告的 20–100×。
   原本擔心「小模型看不到離群現象」的疑慮被否定。
2. `per-channel` 反而輸給 `group-64` —— scale 必須**逐 token 適應**，
   這一點在文獻推論階段沒預料到。
3. 誤差隨 hop 累積但**次線性**（14× hop 只放大 2.8× 不一致率）。

**對等價性宣稱的影響**：量化本身就讓約 1–1.8% 的 token 與 fp32 模型不同，
所以等價性必須表述為「與同一條量化流水線相同」。見
[01-architecture.md §3.1](01-architecture.md)。

```bash
python3 spike/quant_sweep.py --outliers              # 離群統計
python3 spike/quant_sweep.py --sweep all             # 完整掃描 + Q3 累積表
```

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
