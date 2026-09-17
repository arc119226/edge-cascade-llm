/**
 * Service worker。
 *
 * 兩個目的：
 *   1. 讓這個 app 離線也能開（預快取程式碼本體）
 *   2. 讓模型權重的快取由 service worker 統一管理，
 *      這樣重新整理分頁、甚至關掉再開，都不用重下載數 GB
 *
 * 權重「不」放進預快取清單 —— 那是幾 GB 的東西，
 * 必須等使用者明確按下開始才下載。這裡只做執行期快取。
 */

const SHELL_CACHE = 'edge-cascade-shell-v1';
const WEIGHT_CACHE = 'edge-cascade-weights-v1'; // 與 src/cache.js 同名，共用同一份

self.addEventListener('install', (event) => {
  event.waitUntil((async () => {
    const cache = await caches.open(SHELL_CACHE);
    try {
      const shell = await (await fetch('./shell.json')).json();
      await cache.addAll(shell);
    } catch {
      // 預快取失敗不該讓安裝失敗；線上仍然可用，只是沒有離線能力
    }
    await self.skipWaiting();
  })());
});

self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    const keep = new Set([SHELL_CACHE, WEIGHT_CACHE]);
    for (const name of await caches.keys()) {
      if (!keep.has(name)) await caches.delete(name);
    }
    await self.clients.claim();
  })());
});

self.addEventListener('fetch', (event) => {
  const { request } = event;
  if (request.method !== 'GET') return;

  const url = new URL(request.url);
  const isWeight = url.pathname.includes('/model/');

  if (isWeight) {
    // 權重：快取優先。它們不會變，而且重抓的代價極高。
    event.respondWith((async () => {
      const cache = await caches.open(WEIGHT_CACHE);
      const hit = await cache.match(request);
      if (hit) return hit;
      const res = await fetch(request);
      if (res.ok) cache.put(request, res.clone()).catch(() => {});
      return res;
    })());
    return;
  }

  // 程式碼：網路優先、失敗才回快取。
  // 這樣使用者總是拿到最新版，但離線時仍然打得開。
  event.respondWith((async () => {
    try {
      const res = await fetch(request);
      if (res.ok && url.origin === self.location.origin) {
        const cache = await caches.open(SHELL_CACHE);
        cache.put(request, res.clone()).catch(() => {});
      }
      return res;
    } catch {
      const hit = await caches.match(request);
      if (hit) return hit;
      throw new Error('離線且沒有快取');
    }
  })());
});
