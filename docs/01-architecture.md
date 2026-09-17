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

### 3.1 等價性保證（精確表述）

投機解碼的等價性是**相對於「同一條流水線」**，不是相對於原始 fp32 模型：

| 取樣模式 | 保證 | 機制 |
|---|---|---|
| greedy | 與**同一條流水線**自迴歸執行的結果逐 token 相同 | 只接受 draft 與 target argmax 相符的前綴 |
| temperature sampling | 與**同一條流水線**的分布等價 | rejection sampling 修正（Leviathan / Chen 式） |

**這個區別在 M4 之後變得非必要不可**。實測顯示激活值量化本身就會讓
約 0.7%（3 hop）到 1.8%（14 hop）的 token argmax 與 fp32 模型不同。
所以：

```
fp32 單機模型  ──（激活值量化，M4 實測 ~1% token 不同）──>  量化後的流水線
                                                               │
                                          （投機解碼，此處逐 token 精確）
                                                               ▼
                                                        實際輸出
```

投機解碼那一段是**精確無損**的；有損的是量化那一段。
把兩者混為一談會讓「lossless」這個詞失去意義。

> M5 的驗收條件因此是：
> 「greedy 模式下，投機解碼的輸出與**同一條量化流水線**自迴歸執行的結果逐 token 相同」。
> 不是「與 fp32 模型相同」—— 那個在有量化的前提下做不到，且與投機解碼無關。

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

### 4.2 量化格式（M4 已實測敲定）

**採用 `per-channel + 前 3% 通道保留 fp16`**。

> ⚠️ **本節原本寫「線路成本約 8.24 bits/值」，那是錯的 —— 它只算了 int8 本體。**
>
> 漏掉的兩項都不是可選的，必須跟著每一則訊息走：
>
> 1. **scale 表**。`perChannel()` 的 `scale[c]` 是對「這則訊息裡的這 K 個 token」
>    取 `max|x[t][c]|` 算出來的，所以它跟著資料走 —— 接收端推不出來，也不能快取。
> 2. **離群 channel 的索引表**。離群集合每則訊息重挑一次，所以也得送。
>
> 真實成本見 §4.5 的表，由 [`bench/wire_size.py`](../bench/wire_size.py) 產生。
> K=8 時是 **10.24 bits/值**，不是 8.24 —— 差 24%。
>
> 這個漏算會讓方案排名反轉：把中繼資料算進去之後，group-64 在任何實際會用到的
> K 底下都比較省。所以線路格式改成**兩個方案都實作、由 frame header 的
> `schemeId` 決定**，等 M3 量到真實位元組與延遲再選預設值。

完整數據：[`docs/data/quant-results.md`](data/quant-results.md)，
可用 `python3 spike/quant_sweep.py --sweep all` 重現。

SmolLM2-135M、4 段（3 hop）實測：

| 方案 | 線路位元 | PPL 退化 | argmax 一致 |
|---|---|---|---|
| per-tensor | 8.0 | **+30444%** | **4.30%** |
| per-token | 8.0 | +1.87% | 94.09% |
| per-channel | 8.0 | +6.23% | 85.01% |
| group-64 | 8.0 | +0.48% | 97.56% |
| **per-ch + outlier 3%** | **8.2** | **+0.03%** | **99.32%** |

三個實測結論：

1. **per-tensor INT8 確實不可用**，而且比文獻描述的更糟 —— PPL 從 13.8 暴增到 4208。
2. **scale 必須逐 token 適應。** `per-channel` 的 scale 形狀是 `[1,1,d]`，
   跨所有 token 共用，所以只要有一個 token 在該 channel 出現巨值，
   那個 channel 的 scale 就對**所有** token 都被撐大 —— 這就是為什麼
   per-channel（+6.23%）反而輸給 group-64（+0.48%，scale 形狀 `[1,T,g,1]`）。
   這一點違反直覺，是實測才發現的。
3. **離群通道保 fp16 的投報率極高**：多 3% 頻寬換來 argmax 一致率從 85% 拉到 99.3%。

若不想處理混合精度，`group-64` 是最好的純 int8 選項（+0.48%）。

> 「零額外頻寬」也是錯的：group-64 每個 token、每 64 個 channel 各要一個 scale，
> 所以是 8.25 bits/值（fp16 scale）而不是 8.0。它仍然是最省的選項，
> 但不是因為原本說的理由。

### 4.3 量化誤差會隨 hop 數累積（次線性）

M1 已確認**未量化**時切分本身完全無損（2/4/8/15 shard 誤差同為 7.725e-05）。
加上量化後，誤差確實會累積，但是次線性的：

| 方案 | 1 hop | 3 hop | 7 hop | 14 hop |
|---|---|---|---|---|
| per-ch + outlier 3% | 99.37% | 99.32% | 98.58% | 98.24% |
| group-64 | 98.14% | 97.56% | 96.53% | 95.80% |

hop 數增加 14 倍，最佳方案的**不一致率**只放大約 2.8 倍。

> **對拓撲規劃的影響**：P 越大品質越差。這與 §2「P 要盡量少」的結論同向 ——
> 固定開銷與量化誤差都懲罰大 P。

### 4.4 傳輸層

WebRTC DataChannel（reliable + ordered）。以下每一項都在容器裡用兩個
Playwright browser context 實測過，不是照抄規格書：

**單則訊息上限是協商出來的，不是常數。** 讀 `pc.sctp.maxMessageSize`，
不要寫死 256 KiB。實測 Chromium 對 Chromium 是 262144；兩個 Firefox 之間是
1073741823；對方若沒在 SDP 裡宣告 `a=max-message-size`，依 RFC 8841 是 65536。

**超過上限不會丟例外，會非同步殺掉整個 channel。** 實測：單次 `send()` 一個
3,145,728 位元組的訊息**沒有**丟 TypeError（與 W3C 的 send() 演算法描述不符），
正常返回，然後非同步觸發
`error { name: 'OperationError', errorDetail: 'data-channel-failure' }` 並關閉，
0 位元組送達。**這是這一層最危險的失敗模式**，因為它看起來像對方斷線。

**但 `bufferedAmount` 堆高本身是安全的。** 同樣實測：用 16 KiB 分塊、完全不做
背壓，堆到 3,145,728 沒事，繼續堆到 16,777,216 也沒事 —— 到 16 MiB 時
`send()` 丟出**可以 catch 的** `OperationError`，channel 仍是 `open`。
（先前根據 dcSCTP 原始碼推測的「2 MB 送出佇列會殺掉 channel」**沒有重現**。
推測不能當實測用，這裡記下來。）

所以正確的規則只有一條：**永遠不要送出單一則大於 `maxMessageSize` 的訊息。**
分塊之後背壓是為了控制記憶體，不是為了避免 channel 被殺。

**分塊用 16384 位元組**，不是用 `maxMessageSize`。理由不是訊息上限，是
head-of-line blocking —— RFC 8260 的 ndata 在 Chrome 是關閉的、在 Firefox
未實作（bugzilla 1381145），所以一則大訊息在途中會獨佔整個 association。
16 KiB 是 libp2p、webrtc samples、PeerJS 各自收斂到的同一個值。

**不要傳 `maxRetransmits` 或 `maxPacketLifeTime`。** 部分可靠傳輸在 dcSCTP
是啟用的，誤設會讓分片被靜默丟棄而不是報錯 —— 掉一塊就整個張量壞掉。
兩個一起傳會丟例外。

**兩邊都要明確設 `binaryType = 'arraybuffer'`**，包含 `ondatachannel` 收到的
那一個 —— 各家引擎的預設值歷史上並不一致。

**M3 用不到 STUN 也用不到 TURN。** 同一台機器的兩個分頁走 loopback candidate，
同一個區域網路的兩台裝置走 host / mDNS candidate，全程不接觸任何第三方。
公網穿透與 TURN 是 M6 的事。

> TURN 是一個中心化元件。架構文件不應宣稱「完全去中心化」。
> 但也不該把 TURN 寫成每個里程碑都需要 —— M3 不需要。

---

### 4.5 Frame 格式（M3 制定）

> 本節之前不存在。§4.1–4.4 講的是「傳什麼、多大、用什麼傳」，
> 但沒有任何位元組層級的定義 —— 沒有 header、沒有版本、沒有序號、
> 沒有分片框架、沒有端序、沒有訊息型別、沒有回程路徑的定義。
> M3 不是在實作一個既有格式，是在制定一個。

#### 4.5.1 兩層結構

| 層 | 負責 | 實作 |
|---|---|---|
| **frame** | 一則語意完整的訊息（一次交接的激活值，或回程的 logits）| `web/src/wire.js` |
| **chunk** | 把 frame 切成 16 KiB 送出、在對面重組 | `web/src/dcframe.js` |

分片在 frame **底下**，所以 frame 層完全不必知道 DataChannel 的存在，
可以用假的 channel 單元測試。

#### 4.5.2 Frame header（固定 32 位元組，little-endian）

端序一律 little-endian，明確寫死，不依賴平台。

| 位移 | 大小 | 欄位 | 說明 |
|---|---|---|---|
| 0 | 4 | `magic` | `0x314C4345`（`'ECL1'`）|
| 4 | 1 | `version` | 目前是 1 |
| 5 | 1 | `msgType` | 0=激活值 1=logits 2=控制 |
| 6 | 1 | `schemeId` | 0=none 1=per-channel 2=group-64 3=per-ch+outlier |
| 7 | 1 | `flags` | bit0：scale 用 fp16（0=fp32）。其餘保留為 0 |
| 8 | 4 | `roundId` | 這是第幾次 traversal |
| 12 | 1 | `stageIndex` | 產生這則訊息的 shard 序號 |
| 13 | 1 | — | 保留，0 |
| 14 | 2 | `dModel` | |
| 16 | 2 | `qLen` | 這則訊息裡有幾個 token 位置（即 K）|
| 18 | 4 | `startPosition` | 第一個 token 的**絕對**位置 |
| 22 | 2 | `scaleCount` | scale 區塊有幾個值 |
| 24 | 2 | `outlierCount` | 離群索引區塊有幾個值 |
| 26 | 4 | `payloadLen` | header 之後的位元組數 |
| 30 | 2 | — | 保留，0（湊滿 32，維持 4 位元組對齊）|

`scaleCount` 與 `outlierCount` **必須明確帶在 header 裡**，不能讓接收端從
`dModel` 與某個約定的 `outlierFrac` 反推。理由：離群比例是呼叫端參數
（`quant.js` 是 0.03、Python 版是 0.01），而且兩個方案的 scale 數量規則不同
（per-channel 是 `d`、group-64 是 `K × ceil(d/64)`）。少了這兩個欄位，
接收端找不到 payload 的邊界。

#### 4.5.3 `startPosition` / `qLen`：M3 用不到，但現在就要有

M3 沒有 KV cache，所以每次 traversal 都從位置 0 重算，`startPosition` 恆為 0。
**還是要送。** 成本是每則訊息 6 個位元組；不送的代價是 M5 要改協定版本。

這正是 Petals 的作法：client 把 `start_from_position` 放進**每一則**推論請求的
metadata，server 據此設定 `prefix_length`
（`inference_session.py:134-135`、`block_functions.py:163-168`）。
回滾因此是 O(1)、冪等、可重放，而且不需要另外的控制訊息。

現在的程式碼在每個節點各自重算 `arange(0..seqLen)` 當 position_ids
（`web/src/runner.js:271-272`）。那只在「每次都從 0 重算」時才正確 ——
KV cache 或投機解碼一落地，各節點就會靜默地對不上。所以位置要**跟著 frame 走**。

> Petals 自己的那個斷言寫錯了：`block_functions.py:165-167` 斷言的是
> 一個單元素 tuple，永遠為真，所以 client 送出往前跳的位置會靜默汙染快取。
> 我們要加的是真正的單調性檢查。

#### 4.5.4 Payload 佈局

區塊順序刻意讓**所有 ≥2 位元組的區塊排在前面、1 位元組的 int8 本體排最後**，
這樣 `Uint16Array` / `Float32Array` 視圖都能直接建在對齊位置上，不用複製。

本體一律 token-major（t 外層、c 內層），對應 `x[t * dModel + c]`。

| schemeId | payload |
|---|---|
| 0 `none` | fp32 本體：`d × K × 4` |
| 1 `per-channel` | scale 區塊 `d × S`，然後 int8 本體 `d × K` |
| 2 `group-64` | scale 區塊 `K × ceil(d/64) × S`，然後 int8 本體 `d × K` |
| 3 `per-ch+outlier` | 離群索引 `nOut × u16`，scale 區塊 `(d − nOut) × S`，fp16 離群本體 `nOut × K × 2`，最後 int8 本體 `(d − nOut) × K` |

S 是 scale 的位元組數（flags bit0：fp16=2、fp32=4）。

方案 3 的 int8 本體只含**非離群** channel，依 channel 索引遞增排列；
離群 channel 的值放在 fp16 區塊，順序與索引區塊一致。
接收端用索引區塊重建一張 `isOutlier` 表，然後逐 channel 還原。

#### 4.5.5 規範性語意：用「送出去的那個 scale」量化

編碼端**必須**先把 scale 轉成它要送出的精度（預設 fp16），**再**用那個值做量化。

否則 `decode(encode(x))` 不會等於編碼端自己算的結果 —— 編碼端用 fp32 scale
量化、解碼端用 fp16 scale 還原，兩邊差一個 scale 的捨入誤差。
先轉再量化就沒有這個問題，而且解碼端不需要知道原始 fp32 scale。

這條規則讓 `wire.js` 自洽。它與 `quant.js` 的關係則是：
**`quant.js` 是 M4 的品質模擬器，不是線路格式的實作。**
`quant.js` 的 `SCHEMES.fn` 回傳 `Float32Array`（量化→反量化的往返），
從來沒有產生過任何位元組。兩者的綁定測試是「fp32 scale 時逐位元相等、
fp16 scale 時相對誤差 < 2⁻¹⁰」，而不是無條件相等。

#### 4.5.6 Chunk 子標頭（8 位元組，little-endian）

每個 chunk 前面加：

| 位移 | 大小 | 欄位 |
|---|---|---|
| 0 | 4 | `messageId` u32 |
| 4 | 2 | `chunkIndex` u16 |
| 6 | 2 | `chunkCount` u16 |

所以每個 chunk 實際送出 8 + 最多 16376 位元組，總長不超過 16384。

#### 4.5.7 回程：最後一段不在頭節點時送什麼

**送完整的 fp32 logits，分塊 + 背壓。**

M3 的驗收標準要比對所有位置的 logits，所以需要完整 logits ——
回傳 top-k 會讓驗收測試變弱。而這也正好是壓力測試要的：
`vocab_size` 49152 × seq 16 × 4 位元組 = **3,145,728 位元組**，
遠超過協商出來的 262144 上限，所以它一定要走分塊路徑。
直接 `send()` 會照 §4.4 描述的方式非同步殺掉 channel。

預設拓撲讓頭節點同時持有第一段與最後一段（見 §4.5.8），
所以 logits 根本不過線。回程路徑只在壓力測試的拓撲下才會用到 ——
但它必須存在，而且 `msgType = 1` 現在就要佔位，
這樣之後改成 top-k 表示法不需要升版本。

#### 4.5.8 M3 的拓撲

4 段分給 2 個分頁：**頭節點持有 {0, 3}，對等節點持有 {1, 2}。**

§1 已經允許頭節點同時承擔一段。這個切法給出真正的環狀 A→B→A、
兩次跨線，而且兩次跨的都是小的激活值；logits 在頭節點內部產生，不過線 ——
與 §4.1「logits 直接在頭節點用」一致。

另外保留一個設定旗標 `{0,1} / {2,3}`，**只給壓力測試用**：
那個切法會讓 3 MiB 的 fp32 logits 真的跨線，把分塊與背壓路徑跑到。

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
