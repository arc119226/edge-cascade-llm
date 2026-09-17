/**
 * 安裝匯出模型所需的 Python 套件。
 *
 * 為什麼要包成一支腳本，而不是叫使用者自己打 pip：
 *
 * 1. **指令名稱在各平台不同** —— Windows 沒有 `python3`。
 * 2. **`pip` 在 Windows 11 上可能被擋** —— Smart App Control 會封鎖未簽章的
 *    執行檔，而 `pip.exe` 正是 pip 產生的未簽章外殼，訊息還會誤導成
 *    「已被貴組織的 Device Guard 原則封鎖」，讓人以為電腦被公司管控。
 *    走 `python -m pip` 就沒事，因為 `python.exe` 有簽章。
 *
 * 使用者不需要知道上面任何一件事，跑 `npm run setup` 就好。
 */

import path from 'node:path';
import { findPython, runPython, describe } from './python.mjs';

const PACKAGES = ['torch', 'transformers', 'onnx', 'onnxruntime', 'onnxscript'];

const repo = path.resolve(import.meta.dirname, '..', '..');
const py = findPython();

console.log(`使用 ${describe(py)} 安裝：${PACKAGES.join(' ')}`);
console.log('（走 python -m pip，而不是 pip.exe —— 後者在 Windows 11 上可能被 Smart App Control 擋下）\n');

try {
  runPython(py, ['-m', 'pip', 'install', ...PACKAGES], repo);
} catch {
  // execFileSync 在子程序失敗時會拋，但 pip 自己的錯誤訊息已經印出來了，
  // 再把 stack trace 疊上去只會蓋掉真正有用的資訊。
  console.error('\n套件安裝失敗。上面 pip 的輸出通常已經說明原因。');
  console.error('常見狀況：');
  console.error('  - 磁碟空間不足（torch 解壓後約 1 GB）');
  console.error('  - 網路或代理伺服器問題');
  console.error(`  - Python 版本太舊：請確認 ${describe(py)} --version 是 3.10 以上`);
  process.exit(1);
}

console.log('\n✓ 套件安裝完成。接下來執行：npm run deploy');
