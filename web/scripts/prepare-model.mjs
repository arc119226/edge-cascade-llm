/**
 * 把 spike/export_shards.py 匯出的 shard 放進 dist/model/ 供瀏覽器載入，
 * 並把超過 Cloudflare 單檔上限的檔案切成片段。
 *
 * 為什麼一定要切：Workers 與 Pages 的單檔上限都是 25 MiB，而
 *   - ONNX 的 external data 旁檔動輒上百 MB
 *   - 即使量化成 int4 也不夠（實測最大一段仍有 29 MB），
 *     因為 MatMulNBits 只量化 MatMul，embedding 不在範圍內
 * 所以分塊是必要的，不是最佳化。
 *
 * 模型檔不進版控（.gitignore 有擋），所以測試與本機開發前要先跑這支。
 */
import { cp, mkdir, readdir, writeFile, stat, open } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { execFileSync, spawnSync } from 'node:child_process';
import path from 'node:path';

/**
 * 找出這台機器上的 Python 指令。
 *
 * 不能寫死 `python3` —— Windows 沒有這個指令。它只有 `python`，
 * 而且如果使用者沒把 Python 加進 PATH，還有官方安裝程式附的
 * Python Launcher (`py -3`) 可以用。
 *
 * 另外要小心 Windows 的 Microsoft Store 轉址 stub：系統預裝了一個
 * 叫 python.exe 的東西，執行它會打開市集而不是跑 Python。
 * 所以不能只看「指令存在不存在」，要真的執行一次確認它會回報版本。
 */
function findPython() {
  const candidates = [
    ['python3', []],
    ['python', []],
    ['py', ['-3']],   // Windows Python Launcher
  ];
  for (const [cmd, prefix] of candidates) {
    const r = spawnSync(cmd, [...prefix, '-c', 'import sys; print(sys.version_info[0])'], {
      encoding: 'utf8',
      windowsHide: true,
    });
    // 市集 stub 會回非零離開碼或印不出東西，這裡一併濾掉
    if (r.status === 0 && r.stdout.trim() === '3') return { cmd, prefix };
  }
  console.error(
    '找不到可用的 Python。\n' +
    '  需要 Python 3 才能匯出模型。請先安裝並確認它在 PATH 上：\n' +
    '    Windows  https://www.python.org/downloads/ （安裝時勾選 "Add python.exe to PATH"）\n' +
    '    macOS    brew install python\n' +
    '    Linux    用你的套件管理員安裝 python3\n' +
    '  已嘗試過的指令：python3、python、py -3',
  );
  process.exit(1);
}

/** 用偵測到的 Python 跑一支腳本。 */
function runPython(py, args, cwd) {
  execFileSync(py.cmd, [...py.prefix, ...args], { stdio: 'inherit', cwd, windowsHide: true });
}

// 20 MiB 而非 25 MiB：留餘裕給傳輸編碼與未來可能的上限調整。
const CHUNK_BYTES = 20 * 1024 * 1024;

const root = path.resolve(import.meta.dirname, '..');
const repo = path.resolve(root, '..');
const srcDir = process.env.SHARD_DIR ?? path.join(repo, 'out');
const outDir = path.join(root, 'dist', 'model');

if (!existsSync(path.join(srcDir, 'manifest.json'))) {
  const py = findPython();
  console.log(`找不到 ${srcDir}/manifest.json，改用 ${py.cmd} 匯出一份…`);
  runPython(py, [
    path.join(repo, 'spike', 'export_shards.py'),
    '--model', process.env.MODEL ?? 'HuggingFaceTB/SmolLM2-135M',
    '--shards', process.env.SHARDS ?? '4',
    '--dtype', process.env.DTYPE ?? 'int4',
    '--out', srcDir,
    '--seed', '42',
  ], repo);
  // native.bin 由 verify_shards.py 產生 —— 瀏覽器端測試要拿它當對照組
  runPython(py, [path.join(repo, 'spike', 'verify_shards.py'), '--dir', srcDir], repo);
}

/** 把一個檔案切成 <= CHUNK_BYTES 的片段，回傳片段數（1 表示沒切）。 */
async function splitFile(src, destDir, name) {
  const size = (await stat(src)).size;
  if (size <= CHUNK_BYTES) {
    await cp(src, path.join(destDir, name));
    return { parts: 1, size };
  }

  const parts = Math.ceil(size / CHUNK_BYTES);
  const fh = await open(src, 'r');
  try {
    for (let i = 0; i < parts; i++) {
      const len = Math.min(CHUNK_BYTES, size - i * CHUNK_BYTES);
      const buf = Buffer.allocUnsafe(len);
      await fh.read(buf, 0, len, i * CHUNK_BYTES);
      await writeFile(path.join(destDir, `${name}.part${i}`), buf);
    }
  } finally {
    await fh.close();
  }
  return { parts, size };
}

await mkdir(outDir, { recursive: true });

const chunks = {};
let copied = 0, split = 0, total = 0;

for (const f of await readdir(srcDir)) {
  if (!/\.(onnx|json|bin)$|\.onnx\.data$/.test(f)) continue;
  const { parts, size } = await splitFile(path.join(srcDir, f), outDir, f);
  total += size;
  copied++;
  if (parts > 1) {
    // 索引記錄「這個檔案被切成幾片、原本多大」，瀏覽器端據此重組
    chunks[f] = { parts, size };
    split++;
  }
}

await writeFile(path.join(outDir, 'chunks.json'), JSON.stringify(chunks, null, 2));

console.log(`已複製 ${copied} 個檔案到 dist/model/（合計 ${(total / 1e6).toFixed(0)} MB）`);
if (split) {
  console.log(`  其中 ${split} 個超過 ${CHUNK_BYTES / 1024 / 1024} MiB，已切成片段：`);
  for (const [name, info] of Object.entries(chunks)) {
    console.log(`    ${name}  ${(info.size / 1e6).toFixed(0)} MB -> ${info.parts} 片`);
  }
}
