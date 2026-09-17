# EdgeCascadeLLM 架構規格

> 本文取代 [README](../README.md) 初版的架構描述。修正理由見
> [00-feasibility.md](00-feasibility.md)。

---

## 1. 兩個平行軸

EdgeCascadeLLM 同時用到兩種平行，它們**作用在不同指標上，不可混為一談**：

```
                     深度軸（層）                      位置軸（序列）
                  ─────────────────                ─────────────────
  平行方式        流水線平行                        投機解碼
  依賴性質        嚴格資料依賴，不可打破             因果依賴，可猜測+驗證
  解決什麼        單一裝置塞不下模型                 每 token 要走幾趟流水線
  改善的指標      容量（能跑多大的模型）             延遲（單請求 tok/s）
  空轉怎麼辦      併發請求交錯填滿 bubble            不適用
```

**必須同時做到**，因為它們解決的是不同問題：

- 只做深度軸 → 能跑大模型，但每個 token 要付一整趟 WAN traversal，慢到不能用
- 只做位置軸 → 快，但裝置塞不下模型，等於沒有這個專案

---

## 2. 節點角色

```
┌─ 頭節點 (Head) ─────────────────────────────────────────────┐
│  • tokenizer                                                │
│  • draft model（小模型，1B 級，完整常駐本機）                │
│  • 投機解碼的驗證與取樣邏輯                                  │
│  • 拓撲編排：挑選 stage 節點、監控 RTT、處理 churn            │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌─ Stage 0 ──────┐   ┌─ Stage 1 ──────┐   ┌─ Stage N-1 ────┐
│ embed          │   │ layers[a:b]    │   │ layers[c:]     │
│ + layers[0:a]  │   │                │   │ + norm         │
│                │   │                │   │ + lm_head      │
└────────────────┘   └────────────────┘   └────────────────┘
```

頭節點可以同時兼任某一個 stage。**stage 數 P 要盡量少** —— 每 hop 的固定開銷
（序列化 + dispatch + WebGPU kernel launch）隨 P 線性累積，是主導項之一。
P 的選法：**剛好塞得下模型的最小值**。

---

## 3. 解碼協定

每一輪（round）：

```
1. 頭節點用 draft model 本地產生 γ 個候選 token
   └─ 完全在本機，零網路延遲。這是這個架構最關鍵的一點。

2. 頭節點把 [γ+1 個位置] 送進流水線，走一趟 traversal
   └─ Stage 0 → Stage 1 → … → Stage N-1 → 回頭節點
   └─ 一趟 traversal 就驗證了全部 γ+1 個位置

3. 頭節點比對 draft 與 target 的輸出：
   greedy   → 接受最長的相符前綴
   sampling → 用 rejection sampling 修正，保持分布等價

4. 接受 n 個 token（1 ≤ n ≤ γ+1），回到步驟 1
```

**為什麼 draft model 特別適合 P2P**：draft 在頭節點本機跑，網路延遲為零；
只有驗證要走 P2P。一趟昂貴的 traversal 一次驗證 γ+1 個 token，
正好攤掉 WAN 成本。

### 3.1 等價性保證

| 取樣模式 | 保證 | 機制 |
|---|---|---|
| greedy | **逐 token 與自迴歸相同** | 只接受 draft 與 target argmax 相符的前綴 |
| temperature sampling | **分布等價** | rejection sampling 修正（Leviathan / Chen 式） |

> 這是可測試的性質，不是設計意圖。M5 的驗收條件就是
> 「greedy 模式下輸出與自迴歸逐 token 相同」。

### 3.2 draft-free fallback

沒有可用 draft model 時（例如新模型還沒訓練對應的 draft head），
退化成 **Lookahead decoding**（Jacobi + n-gram pool）。
效能較差（約 1.5× vs 3.5×）但不需要額外訓練。

**不使用 vanilla Jacobi** —— 實測加速僅約 1.02×，撐不起複雜度
（見 [00-feasibility.md §2.3](00-feasibility.md)）。

### 3.3 視窗大小 K 的選擇

K 不是常數，必須綁在**該 stage 最弱節點的 roofline** 上：

$$K \le \min_{\text{stage } i} K^*_i, \quad
K^*_i = \frac{\text{該 stage 權重} / \text{記憶體頻寬}_i}{2 \times \text{該 stage 參數} / \text{FLOPS}_i}$$

手機級節點的 $K^*$ 只有約 6。超過就從「免費」變成線性付出計算成本。
調度器必須在配置拓撲時就決定 K。

---

## 4. 線路格式

### 4.1 每 hop 傳什麼

一個 `[batch, K, d_model]` 的激活值張量，加上位置資訊。

實測值（SmolLM2-135M, d_model=576）：**576 B / token / hop（int8）**，
即 `d_model × 1 byte`。

| 模型 | d_model | K=8 (int8) | K=16 (int8) |
|---|---|---|---|
| 8B | 4096 | 32 KB | 64 KB |
| 32B | 5120 | 40 KB | 80 KB |
| 70B | 8192 | 64 KB | 128 KB |

### 4.2 量化：不能用 per-tensor INT8

激活值有**離群通道**，數值達中位數的 20–100 倍，且在 6–7B 以上模型必然出現
（[LLM.int8()](https://arxiv.org/pdf/2208.07339)）。per-tensor INT8 的 scale 被
離群值撐大後，其餘 99.9% 的值只剩 2–3 個有效位元。

**必須採用**：

- 逐通道 / 分組（group-wise）量化，而非 per-tensor
- 離群通道保留 fp16（LLM.int8 式混合精度）
- 或 SmoothQuant 式把數值範圍搬進權重

M1 已確認**未量化**時切分本身是無損的：固定輸入下 2/4/8/15 個 shard
的誤差完全相同（7.7e-05）。但那是沒有 hop 量化的前提。
加上 int8 量化後每個 hop 都會引入新的誤差，累積行為是未知的 ——
**M4 必須用固定輸入的控制實驗量「誤差 vs 切分數」曲線**。

### 4.3 傳輸層

WebRTC DataChannel（reliable + ordered）。注意：

- 單則訊息上限 **256 KiB** → 大 K 需應用層分片
- 用 `bufferedAmount` 做背壓，不要無上限塞
- NAT 直連會失敗一定比例 → **必須有 TURN fallback**

> TURN 是一個中心化元件。架構文件不應宣稱「完全去中心化」。

---

## 5. 容錯

| 情境 | 處理 |
|---|---|
| stage 節點中途離線 | 切到熱備節點，從最近的 activation checkpoint 續傳 |
| 熱備節點從哪來 | 必須**預先載入權重**。冷節點要下載數 GB，來不及接手 |
| 代價 | 冗餘係數 R=2 → 有效算力利用率 50% |

每節點每分鐘掉線率 5% 時，P=8 有 **34% 的對話會中斷**
（`bench/model.py --churn`）。這是 P 要壓低的另一個理由。

---

## 6. 明確不做的事

- **不宣稱隱私保護**。激活值可反推 prompt，v1 僅限信任網路，文件要誠實標示
- **不做拜占庭容錯驗證**。浮點跨 GPU 不確定性使 bit-exact 驗證不可能；
  金絲雀張量可被節點特判偵測。v1 靠信任網路 + 冗餘計算
- **不做深度平行**（SNLP / MGRIT 式）。目前文獻是有損的、只在單 GPU 驗證過、
  且需要微調模型
- **不做代幣 / 激勵機制 / 鏈上結算**
