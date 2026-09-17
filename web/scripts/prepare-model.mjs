/**
 * 把 spike/export_shards.py 匯出的 shard 放進 dist/model/ 供瀏覽器載入。
 *
 * 模型檔不進版控（.gitignore 有擋），所以測試與本機開發前要先跑這支。
 * 沒有現成的匯出就直接呼叫 Python 產一份。
 */
import { cp, mkdir, readdir } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import path from 'node:path';

const root = path.resolve(import.meta.dirname, '..');
const repo = path.resolve(root, '..');
const srcDir = process.env.SHARD_DIR ?? path.join(repo, 'out');
const outDir = path.join(root, 'dist', 'model');

if (!existsSync(path.join(srcDir, 'manifest.json'))) {
  console.log(`找不到 ${srcDir}/manifest.json，改用 Python 匯出一份…`);
  execFileSync('python3', [
    path.join(repo, 'spike', 'export_shards.py'),
    '--model', process.env.MODEL ?? 'HuggingFaceTB/SmolLM2-135M',
    '--shards', process.env.SHARDS ?? '4',
    '--out', srcDir,
    '--seed', '42',
  ], { stdio: 'inherit', cwd: repo });
}

await mkdir(outDir, { recursive: true });
let n = 0;
for (const f of await readdir(srcDir)) {
  if (!/\.(onnx|json)$|\.onnx\.data$/.test(f)) continue;
  await cp(path.join(srcDir, f), path.join(outDir, f));
  n++;
}
console.log(`已複製 ${n} 個檔案到 dist/model/`);
