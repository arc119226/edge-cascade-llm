/**
 * 找出這台機器上可用的 Python，並提供執行它的輔助函式。
 *
 * 為什麼需要這個模組：
 *
 * 1. **不能寫死 `python3`** —— Windows 沒有這個指令，只有 `python`，
 *    而且如果使用者安裝時沒勾「Add python.exe to PATH」，就只剩下官方
 *    安裝程式附的 Python Launcher (`py -3`)。
 *
 * 2. **Windows 有「假的 Python」** —— 系統預裝了叫 `python.exe` /
 *    `python3.exe` 的 App Execution Alias 轉址 stub。執行它不會跑你已經
 *    裝好的 Python，而是印一段叫你去 Microsoft Store 安裝的訊息。
 *    那段訊息特別誤導人：它叫你去裝一個你明明已經裝好的東西。
 *    所以偵測時不能只看「指令存不存在」，要實際執行並確認它回報版本。
 */

import { execFileSync, spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import path from 'node:path';

/** repo 根目錄下的虛擬環境位置。 */
const VENV_DIR = path.resolve(import.meta.dirname, '..', '..', '.venv');

/**
 * venv 裡的 python 執行檔路徑（各平台不同）。
 * 回傳路徑字串；不保證它存在或能執行 —— 那交給下面的實際執行測試。
 */
export function venvPython() {
  return process.platform === 'win32'
    ? path.join(VENV_DIR, 'Scripts', 'python.exe')
    : path.join(VENV_DIR, 'bin', 'python');
}

export { VENV_DIR };

/** 系統 Python 的候選順序（找不到 venv 時才用）。 */
const SYSTEM_CANDIDATES = [
  ['python3', []],
  ['python', []],
  ['py', ['-3']], // Windows Python Launcher
];

/**
 * 完整的候選順序：venv 優先。
 *
 * 為什麼 venv 要排第一：本專案的套件裝在 repo 的 .venv 裡，
 * 不動使用者的全域 Python。這不是潔癖 —— 實際踩過：
 * 對全域環境跑 `pip install --upgrade torch` 會把使用者原本的
 * CUDA 版 torch 換成 PyPI 的 CPU 版，連帶弄壞依賴它的 torchaudio。
 * 一個專案的安裝腳本沒有資格改動使用者其他專案共用的套件版本。
 */
function candidates() {
  const venv = venvPython();
  return existsSync(venv)
    ? [[venv, []], ...SYSTEM_CANDIDATES]
    : SYSTEM_CANDIDATES;
}

/** Windows 轉址 stub 的特徵字串。 */
function looksLikeStoreStub(output) {
  return /Microsoft Store|App execution aliases|應用程式執行別名/i.test(output ?? '');
}

/**
 * 回傳 `{ cmd, prefix }`，找不到就印出可操作的指引並結束程式。
 *
 * @param {{quiet?: boolean}} opts
 */
export function findPython(opts = {}) {
  let sawStoreStub = false;

  for (const [cmd, prefix] of candidates()) {
    const r = spawnSync(cmd, [...prefix, '-c', 'import sys; print(sys.version_info[0])'], {
      encoding: 'utf8',
      windowsHide: true,
    });
    if (r.status === 0 && r.stdout.trim() === '3') return { cmd, prefix };
    if (looksLikeStoreStub(r.stdout) || looksLikeStoreStub(r.stderr)) sawStoreStub = true;
  }

  if (opts.quiet) return null;

  if (sawStoreStub) {
    // 這個情況值得單獨講：使用者多半「已經裝好 Python 了」，
    // 只是 PATH 上排在前面的是 Windows 的轉址 stub。
    // 照 stub 的訊息去市集再裝一次通常無濟於事。
    console.error(
      '偵測到 Windows 的 App Execution Alias 轉址 stub。\n' +
      '  `python3.exe` / `python.exe` 這兩個指令指向的不是真的 Python，\n' +
      '  而是一個會叫你去 Microsoft Store 安裝的捷徑 —— 即使你已經裝好 Python 了。\n' +
      '\n' +
      '  三個解法擇一：\n' +
      '    1. 用 Python Launcher：確認 `py -3 --version` 跑得起來（官方安裝程式預設會裝）\n' +
      '    2. 把 Python 加進 PATH：重跑安裝程式選 Modify，勾選 "Add python.exe to PATH"\n' +
      '    3. 關掉轉址：設定 → 應用程式 → 進階應用程式設定 → 應用程式執行別名，\n' +
      '       把 python.exe 與 python3.exe 關閉',
    );
  } else {
    console.error(
      '找不到可用的 Python。\n' +
      '  需要 Python 3 才能匯出模型。請先安裝並確認它在 PATH 上：\n' +
      '    Windows  https://www.python.org/downloads/ （安裝時勾選 "Add python.exe to PATH"）\n' +
      '    macOS    brew install python\n' +
      '    Linux    用你的套件管理員安裝 python3\n' +
      '  已嘗試過的指令：.venv 內的 python、python3、python、py -3\n' +
      '\n' +
      '  提示：`cd web && npm run setup` 會建立 .venv 並裝好所有套件。',
    );
  }
  process.exit(1);
}

/** 用偵測到的 Python 跑一支腳本或模組，輸出直接接到終端機。 */
export function runPython(py, args, cwd) {
  execFileSync(py.cmd, [...py.prefix, ...args], { stdio: 'inherit', cwd, windowsHide: true });
}

/** 人看得懂的指令字串，用於訊息輸出。 */
export function describe(py) {
  return [py.cmd, ...py.prefix].join(' ');
}
