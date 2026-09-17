# edge-cascade-llm

> 零安裝、基於 WebRTC 的邊緣裝置 LLM 流水線並行推理引擎

EdgeCascadeLLM 是一個概念性架構，旨在透過瀏覽器原生的流水線並行（Pipeline Parallelism）技術，跨異質邊緣裝置（手機、筆電、桌機）執行大語言模型，進而突破單一裝置的 VRAM 限制。

---

## 專案概要

運行現代基座模型（Foundation Models）需要極高的 VRAM 容量，這是消費級邊緣裝置所欠缺的。EdgeCascadeLLM 透過在對等網路（P2P Network）中動態垂直拆分 Transformer 模型層來解決此瓶頸。

EdgeCascadeLLM 無需將運算委外給集中式雲端服務商，也不需要在本地安裝任何執行檔；而是直接在標準 Web 瀏覽器內部，利用 WebGPU 進行邊緣推理，並透過 WebRTC DataChannel 在裝置之間傳輸低延遲的中間張量（Activation Tensors）。

---

## 核心架構理念

* **瀏覽器原生模型層拆分：** 利用 WebGPU 與 `@mlc-ai/web-llm` 動態切割 Transformer 模型（例如：裝置 A 載入第 1–16 層，裝置 B 載入第 17–32 層）。
* **點對點張量直連傳輸：** 在推理期間，經量化處理的中間張量透過 WebRTC 直接在瀏覽器之間傳輸，完全繞過集中式伺服器。
* **去中心化信令與拓撲編排：** 採用 Nostr Relay 進行無伺服器節點發現與 SDP 握手，並根據裝置 VRAM 與網路 RTT 進行實時網路拓撲動態編排。
* **動態離線容錯機制：** 針對節點隨時離線（Node Churn）的特性，透過上游張量快取與 50ms 級別的斷路熔斷機制，將中間張量無縫重路由至預備節點。
* **無信任帳本與計算驗證：** 引入金絲雀陷阱張量（Canary Tensor Injection）與非同步概率雙檢機制驗證運算完整性，並建立離線優先（IndexedDB）的算力積分帳本。

---

## 概念流水線架構

```
[ 輸入 Prompt ]
       │
       ▼
[ 節點 A：筆電 ]   ─── (WebGPU: 第 1 - 16 層) ───►  [ 中間張量 Activation (INT8) ]
                                                                    │
                                                     (WebRTC P2P DataChannel)
                                                                    │
[ 節點 B：手機 ]   ◄── (WebGPU: 第 17 - 32 層) ─────────────────────┘
       │
       ▼
[ Token 輸出 ]

```

---

## 專案狀態與願景

本儲存庫作為 EdgeCascadeLLM 的核心架構規範，定義了建構無摩擦公用 LLM 算力網路所需的通訊協定、網路拓撲演算法以及 P2P 張量傳輸介面。
