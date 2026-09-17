# 部署到 Cloudflare Workers

`web/` 是一個純靜態站。這份指南從你在 Cloudflare 主控台看到的畫面開始，
一路走到站台上線。

**不需要綁付款方式、不需要 R2、不需要任何額外服務。**

---

## 先看懂一個限制

Cloudflare 的靜態資產有 **單檔 25 MiB** 上限（Workers 和 Pages 都一樣）。
而這個專案有兩種檔案會超過：

| 檔案 | 大小 | 怎麼處理 |
|---|---|---|
| ONNX Runtime 的 WebGPU 執行檔 | 28 MB | 從 **jsDelivr CDN** 載入，不放進部署 |
| 模型權重（ONNX external data） | 最大 89 MB | 建置時**自動切成 20 MiB 的片段**，瀏覽器抓完再拼回去 |

這兩件事建置腳本都已經處理好，你不用手動做什麼。
建置最後會自我檢查，`dist/` 裡若還有超過 25 MiB 的檔案會直接報錯。

---

## 步驟

### 一、先在本機產生模型

模型檔不進版控（太大），所以要先在本機產生一份。

```bash
# 需要 Python 套件
pip install torch transformers onnx onnxruntime onnxscript

cd edge-cascade-llm

# 匯出模型並量化成 int8（約 205 MB）
python3 spike/export_shards.py \
  --model HuggingFaceTB/SmolLM2-135M \
  --shards 4 --dtype int4 --out out/ --seed 42

# 驗證切分正確，並產生瀏覽器測試要用的對照檔
python3 spike/verify_shards.py --dir out/
```

> `--dtype int4` 指的是「走量化路徑」，實際位元數由 `--quant-bits` 決定，
> **預設是 8**。為什麼不用真的 4-bit：實測 SmolLM2-135M 在 4-bit 下
> argmax 只剩 3/16（等於壞掉），8-bit 則有 15/16。
> 小模型對低位元量化特別敏感；7B 以上的模型通常撐得住 4-bit。

### 二、建置

```bash
cd web
npm ci
npm run build
node scripts/prepare-model.mjs
```

`prepare-model.mjs` 會把模型複製到 `dist/model/` 並自動切片。
你應該會看到類似：

```
已複製 12 個檔案到 dist/model/（合計 214 MB）
  其中 4 個超過 20 MiB，已切成片段：
    shard_0.onnx.data  89 MB -> 5 片
```

確認沒有超標的檔案：

```bash
find dist -type f -size +25M      # 應該沒有任何輸出
```

### 三、部署

**方法 A：命令列（最快）**

```bash
npx wrangler deploy
```

第一次會要你登入。部署完成後會印出網址。

**方法 B：接 GitHub 自動部署**

從你截圖的那個畫面開始：

1. **Workers & Pages** → **Create application**
2. 選 **Import a repository**，授權並選 `arc119226/edge-cascade-llm`
3. 建置設定填：

   | 欄位 | 值 |
   |---|---|
   | Root directory | `web` |
   | Build command | `npm ci && npm run build` |
   | Deploy command | `npx wrangler deploy` |
   | Build output directory | `dist` |

4. 儲存並部署

> ⚠️ **方法 B 有個前提**：模型檔不在版控裡，所以 Cloudflare 的建置機器
> 產不出 `dist/model/`。站台會上線但按下「開始量測」會失敗。
>
> 要讓自動部署也能用，得先把模型放到一個公開網址
> （例如 GitHub Release 或任何可以 CORS 讀取的靜態空間），
> 再在量測頁的「模型位置」欄位填那個網址。
>
> **第一次先用方法 A**，模型跟程式一起傳上去，最單純。

### 四、確認

打開 `https://你的站.workers.dev/bench.html`，按「開始檢查」。

你應該會看到：

| 項目 | 期望 |
|---|---|
| WebGPU | **可用**（裝置支援的話） |
| 顯示卡 | 有型號 |
| 多執行緒 | **未啟用** ← 這是正常的，見下方說明 |

然後按「開始量測」。第一次要下載約 200 MB，之後有快取就很快。

---

## 為什麼「多執行緒：未啟用」是正常的

WASM 多執行緒需要 `SharedArrayBuffer`，而那需要頁面處於
**cross-origin isolated** 狀態（要送 COOP/COEP 標頭）。

我們預設不開，因為 ORT 在多執行緒模式下會用 wasm 模組的網址去建 Worker，
而跨來源建 Worker 是受限的 —— 而我們的 wasm 正是從 jsDelivr 來的。

**這對量測幾乎沒有影響**：多執行緒只加速 WASM 後端，
而這個工具要量的是 **WebGPU**，WebGPU 不受影響。

（順帶一提：jsDelivr 的檔案確實有帶 `cross-origin-resource-policy: cross-origin`，
所以 CORP 本身不是問題。問題在 Worker 的建構，這一點還沒實測確認。）

### 真的想要多執行緒

```bash
npm run build -- --cross-origin-isolated
```

這會送 COOP/COEP 標頭，並把 wasm 改成同源 —— 但同源就會撞上 25 MiB 上限。
你需要把超標的 wasm 放到 R2 或其他地方，再用
`--assets-url=https://你的網址/ort/` 指過去。這條路要綁付款方式開 R2，
所以列為選配。

---

## 換一個模型

量測頁上的「模型位置」欄位可以填任何提供這些檔案的網址：

```
manifest.json  reference.json  reference.bin  native.bin  chunks.json
shard_N.onnx   shard_N.onnx.data（或它的 .partN 片段）
```

產生方式同步驟一。**上傳前務必先跑 `verify_shards.py`** ——
切分錯誤在瀏覽器裡很難查，輸出會看起來合理、只是慢慢偏掉。

---

## 常見問題

**部署失敗，說檔案太大**
`dist/` 裡有超過 25 MiB 的檔案。跑 `find dist -type f -size +25M` 看是哪個。
若是模型檔，代表 `prepare-model.mjs` 沒跑到；若是 wasm，代表你用了
`--cross-origin-isolated` 卻沒指定 `--assets-url`。

**`no available backend found` / `Failed to fetch dynamically imported module`**
連不到 jsDelivr。檢查網路，或改用 `npm run build -- --self-host-wasm`
把 wasm 放同源（但那樣就會超過單檔上限，只適合本機）。

**`Module.MountedFiles is not available`**
模型的 `.onnx.data` 旁檔沒上傳完整。那些檔案和 `.onnx` 一樣重要。

**`重組後大小不符`**
某個 `.partN` 片段沒傳完整。重跑 `prepare-model.mjs` 再部署一次。

**量測頁顯示「WebGPU：不可用」**
裝置或瀏覽器不支援。工具仍會用 WASM 跑完並回報，那份資料一樣有用 ——
它告訴我們哪些裝置進不了 WebGPU 這條路。

**手機跑到一半沒反應**
多半是記憶體不足被瀏覽器清掉。把模型切更多段（每段更小）再試。

---

## 本機開發

```bash
cd web
npm ci
npm run build
node scripts/prepare-model.mjs
npm run dev          # http://localhost:8080
```

跑測試（會自動建置並準備模型）：

```bash
npm test
```

測試用 headless Chromium 跑完整條流水線，比對原生 ORT 的輸出。
