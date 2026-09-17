# 部署到 Cloudflare Pages

`web/` 是一個純靜態站，可以直接部署到 Cloudflare Pages。
但有兩個大小限制必須先處理，否則部署會失敗或執行期抓不到檔案。

---

## 先理解這兩個限制

### 1. Cloudflare Pages 單檔上限 25 MiB

onnxruntime-web 的 WebGPU 執行檔（`ort-wasm-simd-threaded.jsep.wasm`）約 **28 MB**，
超過上限，**傳不上 Pages**。

### 2. 模型權重更大

一個 32B 模型切成 4 段，每段約 4 GB。這種東西本來就不該放在 Pages 上。

### 解法：大檔全部放 R2，Pages 只放程式碼

```
Cloudflare Pages  ──  HTML / JS / 小的 wasm（約 32 MB）
Cloudflare R2     ──  大的 wasm + 模型權重（數百 MB 到數 GB）
```

R2 沒有出口流量費，很適合這種「同一份檔案被反覆下載」的用途。

---

## 步驟

### 一、建立 R2 儲存桶

```bash
npx wrangler r2 bucket create edge-cascade-assets
```

到 Cloudflare 主控台把它綁一個公開網域（例如 `assets.你的網域.com`），
或啟用 r2.dev 的公開存取。記下這個網址。

### 二、把大檔上傳到 R2

```bash
cd web
npm ci
npm run build -- --assets-url=https://assets.你的網域.com/ort/
```

建置會把超過 25 MiB 的檔案放到 `dist-assets/`，並印出清單。上傳它們：

```bash
for f in dist-assets/*; do
  npx wrangler r2 object put "edge-cascade-assets/ort/$(basename "$f")" --file "$f"
done
```

### 三、上傳模型權重

先把模型切成 shard：

```bash
cd ..
python3 spike/export_shards.py \
  --model HuggingFaceTB/SmolLM2-135M \
  --shards 4 --out out/ --seed 42
```

上傳：

```bash
for f in out/*; do
  npx wrangler r2 object put "edge-cascade-assets/model/$(basename "$f")" --file "$f"
done
```

### 四、設定 R2 的 CORS

**這一步很容易漏掉，漏了瀏覽器會擋下所有請求。**

因為 Pages 站台啟用了 cross-origin isolation（見下方「為什麼需要 COOP/COEP」），
R2 回應必須帶 `Cross-Origin-Resource-Policy: cross-origin`，否則會被擋。

在 R2 儲存桶的設定裡加上 CORS 規則：

```json
[
  {
    "AllowedOrigins": ["https://你的站.pages.dev"],
    "AllowedMethods": ["GET", "HEAD"],
    "AllowedHeaders": ["*"],
    "ExposeHeaders": ["Content-Length", "Content-Range"],
    "MaxAgeSeconds": 86400
  }
]
```

### 五、建立 Pages 專案

在 Cloudflare 主控台連結這個 GitHub repo，設定：

| 欄位 | 值 |
|---|---|
| Root directory | `web` |
| Build command | `npm ci && npm run build -- --assets-url=https://assets.你的網域.com/ort/` |
| Build output directory | `dist` |

部署完成後，`web/_headers` 裡的標頭會自動生效。

### 六、確認

打開 `https://你的站.pages.dev/bench.html`，按「開始檢查」。應該看到：

- **WebGPU：可用**（如果裝置支援）
- **多執行緒：已啟用** ← 這代表 COOP/COEP 生效了

如果「多執行緒」顯示未啟用，表示 `_headers` 沒生效，速度會慢好幾倍。

---

## 為什麼需要 COOP/COEP

WebAssembly 的多執行緒需要 `SharedArrayBuffer`，而瀏覽器只在
**cross-origin isolated** 的頁面上開放它。要做到這件事必須回這兩個標頭：

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

`web/_headers` 已經寫好了。

**代價**：一旦啟用，所有跨來源資源都必須明確允許被嵌入，
所以 R2 那邊的 CORS 設定是必要的，不是可選的。

沒有多執行緒的話 onnxruntime-web 會自動退回單執行緒 —— 還是能跑，
但在多核裝置上會慢好幾倍。

---

## 本機開發

```bash
cd web
npm ci
npm run build            # 不加 --assets-url，全部放本機
node scripts/prepare-model.mjs
npm run dev              # http://localhost:8080
```

不加 `--assets-url` 時建置會警告有檔案超過 Pages 上限 —— 本機開發可以忽略，
它只是提醒你這份建置不能直接部署。

跑測試：

```bash
npm test
```

會用 headless Chromium 跑完整條流水線，比對未切分模型的輸出。

---

## 換一個模型

`bench.html` 上的「模型位置」欄位可以直接改成任何提供
`manifest.json` + `reference.json` + shard 檔案的網址，例如你另一個 R2 路徑。

要產生新的 shard：

```bash
python3 spike/export_shards.py --model <HF 模型 id> --shards <段數> --out out/ --seed 42
python3 spike/verify_shards.py --dir out/    # 先確認切分正確再上傳
```

> 上傳前務必先跑 `verify_shards.py`。切分錯誤在瀏覽器裡很難查 ——
> 輸出會看起來合理，只是慢慢偏掉。

---

## 常見問題

**部署失敗，說檔案太大**
建置指令漏了 `--assets-url`。加上去重新部署。

**`no available backend found` / `Failed to fetch dynamically imported module`**
R2 上的 wasm 檔沒上傳完整，或 `--assets-url` 網址結尾少了斜線。
檢查 `https://你的assets網址/ort/ort-wasm-simd-threaded.jsep.wasm` 打得開。

**`Module.MountedFiles is not available`**
模型的 `.onnx.data` 旁檔沒有跟著上傳。
那些檔案和 `.onnx` 一樣重要，缺一個 session 就建不起來。

**量測頁顯示「多執行緒：未啟用」**
`_headers` 沒生效。確認 Build output directory 是 `dist`，
且 `dist/_headers` 存在（`npm run build` 會複製過去）。

**手機上跑到一半沒反應**
可能是記憶體不足被瀏覽器清掉。試試段數切多一點（每段更小），
或換小一點的模型。
