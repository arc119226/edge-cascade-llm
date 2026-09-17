# EdgeCascadeLLM

> 把一個太大的 AI 模型拆到好幾台裝置上一起跑。

最強的開源語言模型需要 **35 GB** 記憶體。你的筆電沒有，手機更沒有。
但如果三五台裝置各出一點，湊起來就夠了。

這個專案試的就是這件事：**把模型切成幾段，一台裝置只負責其中幾層，接力算完。**

---

## 這在解決什麼問題

大型語言模型是由很多「層」堆起來的。你的問題從第一層流到最後一層，才會產生答案。
層越多，模型越聰明，但也越佔記憶體。

一個 700 億參數的模型，壓縮過後還是要 35 GB。
那超過幾乎所有消費級裝置的上限 —— 你只能買更貴的機器，或付錢用雲端。

**這個專案提供第三條路**：既然一台裝置放不下整個模型，那就每台放一部分。

```mermaid
flowchart LR
    Q["你的問題"] --> A
    A["裝置 A<br/>第 1-8 層"] -->|中間結果<br/>約 30 KB| B
    B["裝置 B<br/>第 9-16 層"] -->|中間結果<br/>約 30 KB| C
    C["裝置 C<br/>第 17-30 層"] --> R["答案"]

    style Q fill:#1e2a4a,stroke:#263050,color:#e8ecf8
    style A fill:#151b30,stroke:#6ea8ff,color:#e8ecf8
    style B fill:#151b30,stroke:#6ea8ff,color:#e8ecf8
    style C fill:#151b30,stroke:#6ea8ff,color:#e8ecf8
    style R fill:#14331f,stroke:#4ade80,color:#4ade80
```

**關鍵在於：裝置之間傳的不是模型本身，而是「算到一半的中間結果」。**

那只有幾十 KB，一般網路就傳得動。
模型權重只要下載一次就存在本機，之後都不用再傳。

---

## 實際的網路長什麼樣子

```mermaid
flowchart TB
    subgraph net["你信任的裝置們（例如自己的機器 + 朋友的）"]
        direction LR
        H["主控裝置<br/>發問與組裝答案<br/>另外跑一個小模型來猜"]
        N1["筆電<br/>第 1-20 層"]
        N2["桌機<br/>第 21-40 層"]
        N3["筆電<br/>第 41-60 層"]
        N4["平板<br/>第 61-80 層"]
    end

    H -->|WebRTC 直連| N1
    N1 --> N2
    N2 --> N3
    N3 --> N4
    N4 -->|答案送回| H

    S["配對服務<br/>只幫裝置互相找到對方<br/>不碰任何資料"] -.介紹彼此.-> H
    S -.-> N1

    style H fill:#1e2a4a,stroke:#6ea8ff,color:#e8ecf8
    style N1 fill:#151b30,stroke:#263050,color:#e8ecf8
    style N2 fill:#151b30,stroke:#263050,color:#e8ecf8
    style N3 fill:#151b30,stroke:#263050,color:#e8ecf8
    style N4 fill:#151b30,stroke:#263050,color:#e8ecf8
    style S fill:#0d1428,stroke:#97a2c4,color:#97a2c4
```

裝置之間是**直接連線**的（WebRTC），資料不經過中央伺服器。
配對服務只做一件事：幫兩台裝置互相找到對方，像交換電話號碼一樣。

> ⚠️ 但**不是完全沒有伺服器**。有些網路環境下裝置無法直連，
> 必須透過中繼站轉送。我們不會假裝這一點不存在。

---

## 為什麼需要「猜」

把模型拆開有一個代價：**每產生一個字，資料都要跑完整條接力**。
如果有 8 台裝置、每次交接要 50 毫秒，那產生一個字就要花將近半秒。太慢了。

解法是讓主控裝置**先用一個小模型猜幾個字**，再把這幾個字**一次送進接力鏈驗證**：

```mermaid
flowchart TB
    S1["1. 主控裝置用小模型快速猜 8 個字<br/>（在本機，不用連網）"]
    S2["2. 把這 8 個字一次送進接力鏈<br/>（只跑一趟）"]
    S3["3. 大模型檢查：猜對幾個？"]
    S4a["前 5 個猜對<br/>直接採用"]
    S4b["第 6 個猜錯<br/>用大模型的答案取代"]
    S5["回到步驟 1，繼續猜下一批"]

    S1 --> S2 --> S3
    S3 --> S4a --> S5
    S3 --> S4b --> S5

    style S1 fill:#1e2a4a,stroke:#6ea8ff,color:#e8ecf8
    style S2 fill:#151b30,stroke:#6ea8ff,color:#e8ecf8
    style S3 fill:#151b30,stroke:#263050,color:#e8ecf8
    style S4a fill:#14331f,stroke:#4ade80,color:#4ade80
    style S4b fill:#331414,stroke:#f87171,color:#f87171
    style S5 fill:#0d1428,stroke:#97a2c4,color:#97a2c4
```

**猜錯也沒關係** —— 猜錯的部分會被大模型的正確答案取代，
所以最後品質和「一個字一個字慢慢算」是一樣的。猜對就賺到，一趟接力產出好幾個字。

這個技巧叫**投機解碼**（speculative decoding），是既有的成熟技術，不是我們發明的。

---

## 老實說的限制

| 限制 | 說明 |
|---|---|
| **比單機慢** | 如果你的裝置本來就塞得下模型，自己跑一定比較快。這個專案是為了跑**本來完全跑不動**的模型。 |
| **沒有隱私保護** | 中間結果有可能被反推出你問了什麼。目前只適合用在**你信任的裝置之間**。 |
| **不是完全去中心化** | 有些網路環境需要中繼伺服器才連得起來。 |
| **第一次很慢** | 每台裝置要先下載自己負責的那幾層，可能要十幾分鐘。之後有快取就快了。 |
| **還在驗證階段** | 核心假設已經驗證過，但多裝置連線還沒做完。 |

前輩專案 [Petals](https://petals.dev/) 已經證明這條路技術上可行，
但它的公開網路現在幾乎沒人用了。
**真正的難題不是技術，是找不找得到夠多人一起跑。**
所以我們刻意從「自己的幾台裝置」這種小規模開始，而不是一開始就做公開網路。

---

## 目前進度

| 階段 | 在問什麼 | 狀態 |
|---|---|---|
| M0 | 這樣做理論上跑多快？ | ✅ 完成 |
| M1 | 把模型切開，結果還對嗎？ | ✅ **完成，結果完全正確** |
| M4 | 壓縮中間結果會不會失真？ | ✅ 完成，找到可用的壓縮方式 |
| M2 | 在瀏覽器裡跑得起來嗎？ | ✅ 完成，且可部署（真實顯示卡效能待實測） |
| M3 | 多台裝置真的連起來 | ⬜ 下一步 |
| M5 | 加上「猜字」加速 | ⬜ 還沒做 |
| M6–M7 | 自動配對、斷線接手 | ⬜ 還沒做 |

**幾個已經驗證的結果：**

- 把模型切成 2、4、8、15 段，算出來的結果**完全一樣** —— 切分本身不會讓模型變笨
- 中間結果可以壓到約 1/4 大小，品質只掉 **0.03%**
- 但壓縮方法要選對：用錯的方法會讓模型直接壞掉（困惑度從 13.8 暴增到 4208）
- 權重量化到 8-bit 沒問題，但 **4-bit 會讓這個 135M 小模型壞掉**
  （選字正確率從 15/16 掉到 3/16）—— 小模型對低位元特別敏感

---

## 你可以幫的忙

我們需要知道各種真實裝置跑起來是什麼樣子 —— 手機、筆電、不同的顯示卡。
開發環境裡沒有顯示卡，這些數字量不到。

**開啟量測頁面，按兩個按鈕，把結果貼回來就好。**
全程在你的瀏覽器裡執行，不會上傳任何東西。

---

## 自己跑跑看

> **Windows 使用者**：把下面的 `python3` 換成 `python`。
> 如果 `pip` 被 Smart App Control 擋下（訊息寫 "Device Guard"），
> 改用 `python -m pip install ...` 即可，**不需要關掉任何安全設定** ——
> 詳見[部署指南的 Windows 一節](docs/DEPLOY.md#windows-使用者請看這裡)。

```bash
# 估算效能（不用下載模型，純數學）
python3 bench/model.py --model 32b --nodes 4

# 驗證「把模型切開結果還對嗎」
pip install torch transformers onnx onnxruntime onnxscript
python3 spike/export_shards.py --model HuggingFaceTB/SmolLM2-135M \
        --shards 4 --dtype int4 --out out/ --seed 42
python3 spike/verify_shards.py --dir out/

# 在瀏覽器裡跑
cd web && npm ci && npm run build && node scripts/prepare-model.mjs
npm test        # 用 headless 瀏覽器驗證整條流水線
npm run dev     # 開 http://localhost:8080 自己玩

# 部署到 Cloudflare（不需要付款方式，見 docs/DEPLOY.md）
npx wrangler deploy
```

---

## 技術文件

這份 README 刻意寫得淺，完整的技術內容在 `docs/`：

| 文件 | 內容 |
|---|---|
| [部署指南](docs/DEPLOY.md) | 怎麼部署到 Cloudflare Workers（免付款方式） |
| [可行性評估](docs/00-feasibility.md) | 量化模型、實測數據、困難點清單 |
| [架構規格](docs/01-architecture.md) | 解碼協定、線路格式、容錯設計 |
| [開發路線](docs/02-roadmap.md) | 里程碑與驗收條件 |
| [待答問題](docs/03-open-questions.md) | 已知的未知數 |

---

## 授權

見 [LICENSE](LICENSE)。
