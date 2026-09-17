# edge-cascade-llm

> 基於 WebRTC 與投機解碼的 P2P 分散式 LLM 推理引擎
> —— 讓消費級裝置跑得動它本來完全跑不動的模型

EdgeCascadeLLM 把一個大型語言模型垂直切分到多個瀏覽器節點上，
用 WebGPU 執行、用 WebRTC DataChannel 在節點之間傳遞中間狀態張量。
目標是讓**沒有任何單一裝置塞得下的模型**（32B / 70B 級）
能在三五台自有裝置組成的私有網路上跑起來。

**這是一個「容量」專案，不是「速度」專案。**
單機跑得動的模型（8B 以下），切分它沒有意義。

---

## 專案狀態

**早期。核心可行性閘門已通過，但尚未有可用的端到端系統。**

| 里程碑 | 狀態 |
|---|---|
| M0 數值模型 | ✅ 完成 |
| **M1 層切分數值等價性（go/no-go 閘門）** | ✅ **通過** |
| M2 瀏覽器 WebGPU | ⬜ 下一步 |
| M3–M7 | ⬜ 未開始 |

完整路線見 [docs/02-roadmap.md](docs/02-roadmap.md)。

---

## 核心架構

系統同時用到兩種平行，它們**作用在不同指標上**：

```
                     深度軸（層）                      位置軸（序列）
                  ─────────────────                ─────────────────
  平行方式        流水線平行                        投機解碼
  依賴性質        嚴格資料依賴，不可打破             因果依賴，可猜測+驗證
  解決什麼        單一裝置塞不下模型                 每 token 要走幾趟流水線
  改善的指標      容量                              延遲
```

### 為什麼是投機解碼

在 P2P 網路上，每走一趟完整流水線要付
`P × (RTT/2 + 每 hop 固定開銷)` —— 動輒數百毫秒。
所以關鍵不是「讓節點不要空轉」，而是**「一趟 traversal 多產出幾個 token」**。

作法是：頭節點**在本機**用一個小 draft model 猜 γ 個 token（零網路延遲），
然後用**一趟**流水線 traversal 一次驗證全部 γ+1 個位置。
昂貴的 WAN 成本因此被攤掉。

```
[ 頭節點：draft model 本地猜 γ 個 token ]   ← 零網路延遲
                  │
                  ▼
[ Stage 0：embed + 層 0–a ]
                  │  hidden state over WebRTC
                  ▼
[ Stage 1：層 a–b ]
                  │
                  ▼
[ Stage N-1：層 c– + norm + lm_head ]
                  │  logits
                  ▼
[ 頭節點：驗證 → 接受 1..γ+1 個 token ]
```

### 等價性

| 取樣模式 | 保證 |
|---|---|
| greedy | **逐 token 與標準自迴歸相同** |
| temperature sampling | **分布等價**（rejection sampling 修正） |

這是可測試的性質，不是設計意圖 —— 見 [路線圖 M5](docs/02-roadmap.md) 的驗收條件。

詳細規格見 [docs/01-architecture.md](docs/01-architecture.md)。

---

## 誠實的限制

這個專案有幾個**不打算粉飾**的限制：

- **不是完全去中心化。** NAT 直連會失敗一定比例，必須有 TURN relay fallback，
  那是一個中心化元件。
- **沒有隱私保護。** 中間激活值可以反推出原始 prompt。
  v1 僅適用於**互相信任的節點**，不要拿來跑敏感內容。
- **沒有計算真實性驗證。** 浮點運算跨 GPU 的不確定性讓 bit-exact 驗證不可能，
  金絲雀張量可被惡意節點特判偵測。v1 靠信任網路，不試圖解拜占庭問題。
- **比單機慢。** 如果你的裝置塞得下模型，單機跑一定比這個快。
- **冷啟動很慢。** 32B/P=4 每個節點要下載約 4 GB，一般家寬約 18 分鐘。

前身專案 [Petals](https://petals.dev/) 證明了這條路技術上可行
（70B 實測可達數 tok/s），但其公開 swarm 已進入 maintenance mode。
**網路效應才是真正的殺手**，不是技術。
因此 v1 範圍刻意收斂到「私有信任 swarm」。

---

## 快速開始

目前可跑的是數值模型與 M1 spike。

### 效能估算

```bash
python3 bench/model.py --model 32b --nodes 4 --rtt 50 --mbps 20
python3 bench/model.py --sweep k          # 掃描平行視窗，含敏感度分析
python3 bench/model.py --roofline         # 各裝置的免費平行視窗 K*
python3 bench/model.py --churn            # 流水線利用率與節點流失
python3 bench/model.py --validate         # 對照 Petals 實測值校驗模型
```

### M1 spike：驗證層切分的數值等價性

```bash
pip install torch transformers onnx onnxruntime onnxscript

python3 spike/export_shards.py --model HuggingFaceTB/SmolLM2-135M --shards 4 --out out/ --seed 42
python3 spike/verify_shards.py --dir out/
```

實測結果（SmolLM2-135M，30 層，固定輸入 seed=42 / seq=32）：

| 切分數 | logits max abs diff | argmax token 一致 |
|---|---|---|
| 2 shard | 7.725e-05 | 32/32 ✅ |
| 4 shard | 7.725e-05 | 32/32 ✅ |
| 8 shard | 7.725e-05 | 32/32 ✅ |
| 15 shard | 7.725e-05 | 32/32 ✅ |

誤差**與切分數完全無關** —— 殘差全部來自 ONNX 對算子本身的匯出，
切分這個動作是無損的。PyTorch 層級的串接誤差是 0.000e+00。

---

## 文件

| | |
|---|---|
| [00-feasibility.md](docs/00-feasibility.md) | 可行性評估、量化模型、困難點清單 |
| [01-architecture.md](docs/01-architecture.md) | 架構規格、解碼協定、線路格式 |
| [02-roadmap.md](docs/02-roadmap.md) | 里程碑與驗收條件 |
| [03-open-questions.md](docs/03-open-questions.md) | 待答問題與已知未知數 |

---

## 授權

見 [LICENSE](LICENSE)。
