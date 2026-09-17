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

三行指令，**完全不用打任何 Python 指令**。
腳本會自己找到你機器上的 Python（`python3` / `python` / `py -3` 都認得），
也會自己繞過 Windows 的兩個陷阱（Smart App Control、App Execution Alias）。

```bash
cd web
npm ci
npm run setup      # 裝 Python 套件（torch 等，約 1 GB）
npm run deploy     # 建置 + 產生模型 + 部署
```

`npm run deploy` 會依序做三件事，順序寫死在 script 裡所以你不會弄錯：

1. `npm run build` —— 建置網站（會清空 `dist/`）
2. `npm run prepare-model` —— 沒有模型就自動匯出、驗證、切片放進 `dist/model/`
3. `wrangler deploy` —— 部署。第一次會要你登入

第一次跑 `npm run prepare-model` 需要幾分鐘（要下載模型並量化），
之後 `out/` 裡有東西就會直接沿用。

> **`npm run setup` 會在 repo 根目錄建一個 `.venv`，套件裝在那裡面，
> 不會動到你的系統 Python。**
>
> 這一點是踩過坑才改的：早期版本直接對全域環境跑 `pip install --upgrade`，
> 把使用者原本的 **CUDA 版 torch** 換成了 PyPI 的 CPU 版，連帶弄壞
> 依賴它的 torchaudio。一個專案的安裝腳本沒有資格改動你其他專案共用的
> 套件版本 —— 在隔離的 venv 裡升級才是安全的。
>
> 本專案需要 **transformers 5.0 以上**（v4 的 `from_pretrained` 不認得
> `dtype` 參數）。腳本會在版本太舊時直接擋下並告訴你怎麼處理。

### 想自己控制匯出參數

只有在你要換模型或調整切分數時才需要手動跑。
**Windows 請用 `python`，不是 `python3`**（原因見下方 Windows 一節）：

```bash
python3 spike/export_shards.py \
  --model HuggingFaceTB/SmolLM2-135M \
  --shards 4 --dtype int4 --out out/ --seed 42

python3 spike/verify_shards.py --dir out/
```

> `--dtype int4` 指的是「走量化路徑」，實際位元數由 `--quant-bits` 決定，
> **預設是 8**。為什麼不用真的 4-bit：實測 SmolLM2-135M 在 4-bit 下
> argmax 只剩 3/16（等於壞掉），8-bit 則有 15/16。
> 小模型對低位元量化特別敏感；7B 以上的模型通常撐得住 4-bit。

---

## Windows 的兩個陷阱（`npm run setup` 已經幫你避開）

這一節是給想理解發生什麼事、或偏好手動執行的人看的。
照上面三行 npm 指令走的話，不會碰到這兩件事。

### 陷阱一：`pip` 被 Smart App Control 擋下

```
'C:\PythonXXX\Scripts\pip.exe' 已被貴組織的 Device Guard 原則封鎖。
```

這是 **Windows 11 的 Smart App Control**（用 WDAC 實作，所以訊息裡寫 Device Guard）。
在個人電腦上是預設行為，**不代表你的電腦被公司管控**。
它會擋掉未簽章的執行檔，而 `pip.exe` 正是 pip 產生的未簽章 .exe 外殼。

手動的解法是走**已簽章的 `python.exe`**：

```bat
python -m pip install torch transformers onnx onnxruntime onnxscript
```

> ⚠️ **不要為了這件事去關掉 Smart App Control。**
> 它一旦關閉就**無法再開啟** —— 要重灌 Windows 才能恢復。
>
> 想確認是不是它擋的：事件檢視器 →
> `應用程式及服務記錄檔 > Microsoft > Windows > CodeIntegrity > Operational`，
> 事件 3033 / 3077 就是封鎖記錄。

### 陷阱二：`python3` 是假的

```
Python was not found; run without arguments to install from the Microsoft Store,
or disable this shortcut from Settings > Apps > Advanced app settings > App execution aliases.
```

Windows 預裝了叫 `python.exe` / `python3.exe` 的 **App Execution Alias 轉址 stub**。
執行它**不會**跑你已經裝好的 Python，只會叫你去市集再裝一次 ——
這個訊息特別誤導人，因為它叫你去裝一個你明明已經裝好的東西。

手動的解法，三選一：

1. 改用 `python`（Windows 上就是這個名字，不是 `python3`）
2. 用 Python Launcher：`py -3 spike\export_shards.py ...`
3. 關掉轉址：設定 → 應用程式 → 進階應用程式設定 → 應用程式執行別名，
   把 `python.exe` 與 `python3.exe` 關閉

完整的 Windows 手動指令：

```bat
python -m pip install torch transformers onnx onnxruntime onnxscript

python spike\export_shards.py --model HuggingFaceTB/SmolLM2-135M ^
       --shards 4 --dtype int4 --out out\ --seed 42

python spike\verify_shards.py --dir out\
```

（`^` 是 cmd 的換行符號。PowerShell 用 `` ` ``，或直接寫成一行。）

**順帶一提**：Python 3.14 沒問題 —— torch、onnxruntime、onnx 都有對應的
Windows wheel。torch 的 Windows 版是 CPU build，只有約 124 MB。

---

## 另一種部署方式：接 GitHub 自動部署

從 Cloudflare 主控台：

1. **Workers & Pages** → **Create application**
2. 選 **Import a repository**，授權並選你的 repo
3. 建置設定：

   | 欄位 | 值 |
   |---|---|
   | Root directory | `web` |
   | Build command | `npm ci && npm run build` |
   | Deploy command | `npx wrangler deploy` |
   | Build output directory | `dist` |

> ⚠️ **這條路有個前提**：模型檔不在版控裡，Cloudflare 的建置機器**產不出**
> `dist/model/`（它沒有 Python 環境，也不會為了一次建置去下載 torch）。
> 站台會上線，但按下「開始量測」會失敗。
>
> 要讓自動部署也能用，得先把模型放到一個公開、可 CORS 讀取的網址
> （例如 GitHub Release），再在量測頁的「模型位置」欄位填那個網址。
>
> **第一次建議先用 `npm run deploy`**，模型跟程式一起傳上去，最單純。

---

## 部署完成後的確認

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

產生方式見上面「想自己控制匯出參數」。**上傳前務必先跑 `verify_shards.py`** ——
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

**`Python was not found; run without arguments to install from the Microsoft Store`**
你打到 Windows 的 App Execution Alias 轉址 stub 了，不是真的 Python。
最簡單的解法是**不要手動打 Python 指令**，改用 `npm run setup` 與 `npm run deploy` ——
它們會自己找到正確的 Python。要手動跑的話見上面「陷阱二」。

**安裝時出現一堆 `WARNING: The scripts ... are installed in ... which is not on PATH`**
可以忽略。那些是 `torchrun.exe`、`hf.exe`、`transformers.exe` 之類的命令列工具，
本專案一個都沒用到（我們只 import 函式庫，不呼叫它們的 CLI）。

**安裝時出現 `ERROR: pip's dependency resolver ... torchaudio requires torch==X, but you have torch Y`**
如果你是用 `npm run setup`（會裝進 `.venv`），這不會發生。

若你是**手動**對全域環境安裝而看到這個，代表你原本的 torch 被換掉了，
依賴它的套件（torchaudio、torchvision 等）會壞掉。本專案不受影響
（我們沒用到那些），但你其他的 PyTorch 專案會。

想還原原本的版本（把版本號換成你原本的，`+cu128` 表示 CUDA 12.8）：

```bat
python -m pip install --force-reinstall ^
       torch==2.10.0 torchaudio==2.10.0 ^
       --index-url https://download.pytorch.org/whl/cu128
```

> PyPI 上的 Windows torch 是 **CPU 版**（約 124 MB）。CUDA 版要從
> `download.pytorch.org` 裝，體積大得多。本專案只需要 CPU 就夠 ——
> 匯出模型不吃 GPU。

**`transformers 版本太舊`**
機器上有舊版 transformers，而 `pip install` 不會自動升級。
跑 `npm run setup`（已帶 `--upgrade`），或手動
`python -m pip install --upgrade transformers`。

**`找不到可用的 Python`**
腳本試過 `python3`、`python`、`py -3` 都沒找到真的 Python。
Windows 上常見原因是安裝時沒勾「Add python.exe to PATH」——
重跑安裝程式選 Modify 補勾，或確認 `py -3 --version` 跑得起來。

> 若腳本偵測到的是轉址 stub，它會另外給一段專門的說明，
> 而不是這則通用訊息。

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
