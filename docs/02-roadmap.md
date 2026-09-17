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
| M5 | KV cache + 投機解碼 | ⬜ |
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
3. WebGPU 版的 wasm 是 28 MB，**超過 Cloudflare 的 25 MiB 單檔上限**
   （Workers 與 Pages 同一個限制）。解法是兩件事，都不動用額外服務：
   ORT 的 wasm 改從 **jsDelivr** 載入、根本不進部署；模型的 ONNX external data
   由 [`web/scripts/prepare-model.mjs`](../web/scripts/prepare-model.mjs)
   在建置時切成 **20 MiB 的 `.partN` 片段**，瀏覽器端由
   [`web/src/cache.js`](../web/src/cache.js) 抓回來重組。見 [DEPLOY.md](DEPLOY.md)

   > **這裡原本寫的是「把大檔分流到 R2」。那是錯的，而且是會傳染的錯。**
   > [DEPLOY.md](DEPLOY.md) 的硬性前提是**不綁付款方式、不用 R2、不用任何額外服務**
   > —— R2 要綁付款方式，把它寫進路線圖等於把「這次部署怎麼繞過 25 MiB」
   > 升級成專案的長期依賴，而且是一個付費依賴。
   >
   > 為什麼切 20 MiB 而不是頂著 25 MiB 切：留餘裕給傳輸編碼
   > （`prepare-model.mjs:18-19`）。建置最後會自我檢查 `dist/` 裡還有沒有超過
   > 25 MiB 的檔案 —— 所以這條線再被踩破時會**建置失敗**，不是部署到一半才失敗。

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
- 量測真實 per-hop 延遲、以及**每趟 traversal 的端到端延遲**

  > **M3 報延遲，不報 tok/s。** 這一階段根本沒有多步解碼迴圈 ——
  > 沒有 KV cache、沒有投機解碼，一趟 traversal 就是一次前向、就結束了。
  > 硬要把它換算成 tok/s，等於憑空編一個不存在的解碼器出來，
  > 而那個編出來的數字之後還會被 M5 當成基準線去比。
- 正確處理**協商出來的**單則訊息上限（分片）與 `bufferedAmount` 背壓
  —— 上限要讀 `pc.sctp.maxMessageSize`，不寫死 256 KiB
  （實測值因引擎而異，見 [01-architecture.md §4.4](01-architecture.md)）
- **在兩台真實機器、同一個區域網路上手動跑成功一次**：
  不用 STUN、不用 TURN、不用任何伺服器

  > **為什麼「兩台真機」屬於 M3 而不是 M6。** CI 只跑得動同一台機器上的兩個分頁，
  > 那走的是 loopback candidate —— 驗得了 frame 格式、分片與重組，
  > 但完全驗不到 ICE 真的能在兩台裝置之間配對起來。
  > 同一個 LAN 是唯一「不接觸任何第三方服務就能測到真實 ICE」的場景，
  > 所以它必須留在 M3；M6 要處理的是公網穿透與 TURN，是另一回事。
  >
  > 分工寫明白：**CI 負責同機兩分頁，人負責同 LAN 兩台機器。**
  > 後者做不到自動化，但做不到自動化不等於可以不做。

**KV cache 不在 M3，已整個移進 M5**（與投機解碼一起）。

> **理由不是「M3 做不完」，是這兩件事根本不相交。**
> M3 改的是 JS 與 WebRTC；KV cache 改的是 Python/ONNX 匯出端 ——
> `spike/export_shards.py:98-103` 的 `_causal_mask()` 拿 `hidden.shape[1]`
> 造一個**正方形** `[seq, seq]` 的 `torch.triu` 遮罩，結構上就做不了增量解碼；
> 匯出的圖也只吃 `(input_ids|hidden_states, position_ids)`，
> 沒有任何 `past_key_values` 的輸入或輸出。要有 KV cache 就得重新匯出模型。
>
> 綁在一起的代價是**出錯時歸因不了**：輸出對不上了，是 WebRTC 這條路把位元組
> 送壞了，還是 KV cache 的遮罩對不上？這個陷阱這個 repo 已經踩過並記錄過兩次
> —— [Q1](03-open-questions.md) 用「把 4 段收成 2 段」製造 hop 數差異，
> 結果時間差裡同時混了「少幾個 hop」和「少算幾層」；
> [Q3](03-open-questions.md) 用未固定 seed 比較不同切分數，量到的其實是輸入雜訊。
> 兩次都是同一個病：一次動了兩個變因，量出來的數字誰也不能證明。
> **每個里程碑只留一個可被否證的變因。**
>
> 代價是 M3 每趟 traversal 都從位置 0 重算整個序列 —— 慢，但可歸因。
> 線路格式上已經先把位置留好了：frame header 的 `startPosition` / `qLen`
> 現在就送，M3 期間恆為 0（見 [01-architecture.md §4.5.3](01-architecture.md)），
> 這樣 M5 落地時不用升協定版本。

---

## ✅ M4 — 激活值量化

每個 hop 把激活值壓成 int8 以減少傳輸量。
**提前到 M2 之前做**，因為它是 P0 風險且完全可在本地驗證 ——
若 int8 不可用，整個可行性評估要重算。

**產出**：[`spike/quant_sweep.py`](../spike/quant_sweep.py)、
[`spike/quant_schemes.py`](../spike/quant_schemes.py)、
[`spike/corpus.txt`](../spike/corpus.txt)（釘在 repo 的語料，不下載）

**驗收**：PPL 退化 < 1%、傳輸量減半、量出誤差 vs 切分數曲線。**全數達成。**

**結論**：採用 `per-channel + 前 3% 通道 fp16`（**本體** 8.24 bits/值）——
PPL 退化 **0.03%**、argmax 一致 **99.32%**。
純 int8 的最佳選項是 `group-64`（+0.48%）。

> ⚠️ 括號裡的位元數**只算了本體**。把必須跟著每一則訊息走的 scale 表與離群索引表
> 算進去之後，K=8 時真實成本是 **10.24 bits/值**（差 24%）；`group-64` 也不是
> 原本寫的「零額外頻寬」—— 每個 token、每 64 個 channel 各要一個 scale，
> 是 8.25 bits/值。**方案排名會因此反轉。** 真實的位元組帳由
> [`bench/wire_size.py`](../bench/wire_size.py) 產生，表在
> [01-architecture.md §4.2 / §4.5](01-architecture.md)。

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

## ⬜ M5 — KV cache + 投機解碼

頭節點本地 draft model + 流水線一次驗證 γ+1 個 token。
**KV cache 從 M3 移進來**（理由見 M3 一節）：它是匯出端的改動 ——
`spike/export_shards.py` 要改成吃/吐 `past_key_values`，
因果遮罩要從正方形 `[seq, seq]` 改成矩形 `[q_len, kv_len]`。

**驗收**：
- **報兩個 tok/s，分開報，不報「相對 M3 提升幾倍」這一個數字**：

  | # | 組態 | 用途 |
  |---|---|---|
  | (1) | 有 KV cache、**沒有**投機解碼 | KV cache 的驗收基準 |
  | (2) | 有 KV cache、**有**投機解碼 | 投機解碼的驗收 |

  **投機解碼只用 (2) 相對 (1) 來判定，要求 ≥ 2×。**

  > 因為 KV cache 搬進 M5 之後，「相對 M3 提升 ≥ 2×」這個舊條件會被
  > 「不再每步重算整個序列」一項就輕鬆達成，投機解碼即使毫無效果也照樣過關 ——
  > 一個由兩個變因共同產生的數字，證明不了其中任何一個。
  > （何況 M3 報的是每趟 traversal 的延遲、不是 tok/s，本來就不能直接當分母。）
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
