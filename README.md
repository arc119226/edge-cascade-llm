# edge-cascade-llm

> 基於 WebRTC 與雅可比迭代解碼（Jacobi Decoding）的 P2P 平行 LLM 推理引擎

EdgeCascadeLLM 是一個去中心化架構，旨在打破傳統 Transformer 層級運算的強遞歸鎖定。透過結合 WebGPU、WebRTC 點對點網路與**雅可比迭代解碼（Jacobi Decoding）**技術，讓分佈於 P2P 網路中的各個邊緣裝置機能**同時平行計算**模型的不同層級，大幅消滅傳統流水線的等待空窗（Pipeline Bubbles），達成多節點平行解碼 Token。

---

## 專案概要與核心突破

傳統的流水線並行（Pipeline Parallelism）存在嚴格的時間因果依賴：第 $L+1$ 層節點必須等待第 $L$ 層算完才能接手，導致多數裝置在大部分時間處於空轉等待狀態。

EdgeCascadeLLM 引入雅可比迭代解碼（Jacobi Decoding）**機制，將自迴歸 LLM 生成轉化為非線性方程組的**不動點迭代（Fixed-point Iteration）問題：

1. **解耦遞歸依賴：** 系統對未來的 Token 序列給出初始張量猜測，允許負責不同 Transformer 模型層的 P2P 節點 100% 同時開始平行計算。
2. **張量收斂與同步：** 各節點透過低延遲 WebRTC DataChannel 交換中間狀態張量，經過數次快速迭代後，數值自動收斂至標準自迴歸解。
3. **多 Token 一次性平行輸出：** 打破「一字一字傳遞」的限制，透過全網節點的同步計算，實現多 Token 區塊的平行收斂與一次性解碼。

---

## 核心架構理念

* **雅可比平行解碼（Jacobi Parallel Decoding）：** 徹底消除傳統流水線的空調度時間（Pipeline Bubbles）。全網節點針對各自負責的模型層同時進行張量矩陣計算，透過不動點收斂（Fixed-point Convergence）確保輸出結果與標準自迴歸推理完全等價。
* **瀏覽器原生分片（WebGPU & WebLLM）：** 利用 WebGPU 動態將大型語言模型垂直切分至各端側裝置（例如：節點 A 載入第 1–16 層，節點 B 載入第 17–32 層），免去安裝本地原生執行檔。
* **低延遲 P2P 張量同步（WebRTC Tensor Sync）：** 經量化（INT8/INT4）處理的中間狀態張量直接在各 node 之間傳輸，不經過集中式伺服器中轉。
* **去中心化信令與拓撲編排（Nostr Signaling）：** 利用 Nostr Relay 處理無伺服器節點發現與 SDP 握手，並根據節點 RTT 與算力動態編排雅可比收斂週期。
* **動態離線容錯與計算驗證：** 結合上游張量快取（Activation Checkpointing）、金絲雀陷阱張量（Canary Tensor）與離線優先帳本（IndexedDB），確保動態節點離線（Node Churn）時的無縫接管與計算真實性。

---

## 雅可比平行計算流程圖

```
[ 初始張量猜測 / Input Tokens ]
              │
              ├───► [ 節點 A：筆電 (第 1 - 16 層) ]  ──┐ (同時平行計算)
              │                                      │
              └───► [ 節點 B：手機 (第 17 - 32 層) ] ──┤
                                                     │
                                   (WebRTC P2P 張量同步與迭代)
                                                     │
                                                     ▼
                                          [ 狀態檢查：是否收斂？ ]
                                           /                  \
                                     (未收斂)                (已收斂)
                                        /                        \
                            [ 帶入最新張量再迭代 ]      [ 平行輸出 Token 區塊 ]

```

---

## 數學原理概要

傳統自迴歸按序遞歸：


$$y_t = \arg\max P(y_t \mid y_1, y_2, \dots, y_{t-1})$$

EdgeCascadeLLM 採用雅可比平行更新矩陣：


$$y_i^{(k+1)} = \arg\max P\left(y_i \;\middle\vert{}\; y_1^{(k)}, y_2^{(k)}, \dots, y_{i-1}^{(k)}\right) \quad (\forall i \in \{1, \dots, K\})$$

所有節點同時對 $i = 1 \dots K$ 的層與位置進行平行計算，當 $y^{(k+1)} = y^{(k)}$ 時抵達不動點，即刻完成多 Token 平行解碼。

---

## 專案狀態與願景

本儲存庫為 EdgeCascadeLLM 的核心架構規範，定意了基於 WebRTC 與 Jacobi Decoding 的 P2P 平行算力網絡通訊協定、張量收斂演算法與動態拓撲調度介面。
