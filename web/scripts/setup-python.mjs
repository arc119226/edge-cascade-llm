/**
 * 在 repo 根目錄建立虛擬環境，並把匯出模型所需的 Python 套件裝進去。
 *
 * **不會動到你的全域 Python。** 這一點是刻意的，而且是踩過坑學到的：
 *
 * 早期版本直接對全域環境跑 `pip install --upgrade`，結果把使用者原本的
 * CUDA 版 torch（`2.10.0+cu128`）換成了 PyPI 的 CPU 版（`2.14.0`），
 * 連帶弄壞依賴它的 torchaudio。一個專案的安裝腳本沒有資格改動使用者
 * 其他專案共用的套件版本 —— 在隔離的 venv 裡升級才是安全的。
 *
 * 另外順便處理掉 Windows 的兩個陷阱，使用者不需要知道它們存在：
 *   - `python3` 可能是 App Execution Alias 轉址 stub（見 python.mjs）
 *   - `pip.exe` 未簽章，可能被 Smart App Control 擋下 —— 走 `python -m pip` 就沒事
 */

import { existsSync } from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { findPython, runPython, describe, venvPython, VENV_DIR } from './python.mjs';

// transformers 釘在 v5 以上：本專案用 v5 的 API（例如 from_pretrained 的
// `dtype` 參數，v4 只認得 `torch_dtype`）。不寫相容層，因為 v4 的失敗是從
// transformers 內部丟出來的 TypeError，看不出根因，不如直接要求正確版本。
const PACKAGES = [
  'torch',
  'transformers>=5',
  'onnx',
  'onnxruntime',
  'onnxscript',
];

const repo = path.resolve(import.meta.dirname, '..', '..');

// 1. 確保 venv 存在
if (!existsSync(venvPython())) {
  const system = findPython();
  console.log(`建立虛擬環境 ${VENV_DIR}`);
  console.log(`（用 ${describe(system)}；套件會裝在這裡，不會動到你的系統 Python）\n`);
  try {
    runPython(system, ['-m', 'venv', VENV_DIR], repo);
  } catch {
    console.error('\n建立虛擬環境失敗。');
    console.error('  Debian/Ubuntu 上可能要先裝 venv 模組：sudo apt install python3-venv');
    process.exit(1);
  }
} else {
  console.log(`沿用既有的虛擬環境 ${VENV_DIR}\n`);
}

// 2. findPython() 現在會優先挑到 venv 裡的 python
const py = findPython();
if (!py.cmd.startsWith(VENV_DIR)) {
  // 走到這裡代表 venv 存在但不能執行（例如換過 Python 版本、或目錄被搬動過）。
  // 繼續裝下去會污染全域環境，正是這次要避免的事，所以直接停。
  console.error(`虛擬環境存在但無法執行：${venvPython()}`);
  console.error('  請刪掉 .venv 之後重跑 npm run setup。');
  process.exit(1);
}

console.log(`安裝套件：${PACKAGES.join(' ')}`);
console.log('（走 python -m pip，而不是 pip.exe —— 後者在 Windows 11 上可能被 Smart App Control 擋下）\n');

try {
  // --upgrade 在 venv 裡是安全的：影響範圍只有這個專案。
  runPython(py, ['-m', 'pip', 'install', '--upgrade', ...PACKAGES], repo);
} catch {
  // execFileSync 在子程序失敗時會拋，但 pip 自己的錯誤訊息已經印出來了，
  // 再把 stack trace 疊上去只會蓋掉真正有用的資訊。
  console.error('\n套件安裝失敗。上面 pip 的輸出通常已經說明原因。');
  console.error('常見狀況：');
  console.error('  - 磁碟空間不足（torch 解壓後約 1 GB）');
  console.error('  - 網路或代理伺服器問題');
  process.exit(1);
}

console.log('\n✓ 套件安裝完成（裝在 .venv，你的系統 Python 未被更動）。');
console.log('  接下來執行：npm run deploy');
