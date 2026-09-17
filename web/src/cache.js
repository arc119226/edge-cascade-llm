/**
 * 權重快取。
 *
 * 模型 shard 動輒數百 MB。沒有快取的話每次重新整理都要重下載一次，
 * 在一般家寬上 32B 模型每個節點要 4 GB / 約 18 分鐘（見 bench/model.py --cold-start）。
 * 這是這個專案最大的採用門檻之一，所以快取不是優化，是必要功能。
 *
 * 用 Cache API 而不是 IndexedDB：
 *   - Cache API 直接存 Response，不需要把整個 ArrayBuffer 讀進記憶體再寫入
 *   - 它是 service worker 的原生搭檔，離線時同一份快取可以直接服務
 *   - 配額上限通常和 OPFS 同一個池子，但 API 簡單得多
 */

const CACHE_NAME = 'edge-cascade-weights-v1';

/** 快取是否可用（某些情境下會被停用，例如無痕模式或第三方 iframe）。 */
async function openCache() {
  if (typeof caches === 'undefined') return null;
  try {
    return await caches.open(CACHE_NAME);
  } catch {
    return null; // 存取被拒就直接走網路，不要讓整個流程掛掉
  }
}

/**
 * 抓一個 shard 檔案，優先走快取。
 *
 * @param {string} url
 * @param {{optional?: boolean, onProgress?: Function}} opts
 *        optional=true 時，404 會回傳 null 而不是拋錯
 *        （用於 .onnx.data 旁檔 —— 小模型沒有這個檔是正常的）
 * @returns {Promise<Uint8Array|null>}
 */
export async function fetchShard(url, opts = {}) {
  const cache = await openCache();

  if (cache) {
    const hit = await cache.match(url);
    if (hit) {
      opts.onProgress?.({ url, cached: true });
      return new Uint8Array(await hit.arrayBuffer());
    }
  }

  const res = await fetch(url);
  if (!res.ok) {
    if (opts.optional && res.status === 404) return null;
    throw new Error(`抓取 ${url} 失敗：HTTP ${res.status}`);
  }

  // 先放進快取再讀 body：cache.put 會消耗掉 response，所以要 clone
  if (cache) {
    try {
      await cache.put(url, res.clone());
    } catch {
      /* 配額滿了就算了，不影響這次執行 */
    }
  }

  opts.onProgress?.({ url, cached: false });
  return new Uint8Array(await res.arrayBuffer());
}

/** 目前快取了多少位元組，以及有幾個檔案。 */
export async function cacheStats() {
  const cache = await openCache();
  if (!cache) return { available: false, files: 0, bytes: 0 };

  const keys = await cache.keys();
  let bytes = 0;
  for (const req of keys) {
    const res = await cache.match(req);
    if (!res) continue;
    const len = res.headers.get('content-length');
    // 沒有 content-length（例如 chunked 回應）時只好讀出來算
    bytes += len ? Number(len) : (await res.arrayBuffer()).byteLength;
  }
  return { available: true, files: keys.length, bytes };
}

/** 清掉快取的權重。使用者換模型或要釋放空間時用。 */
export async function clearCache() {
  if (typeof caches === 'undefined') return false;
  return caches.delete(CACHE_NAME);
}

/** 瀏覽器願意給這個站多少儲存空間。用來在下載前先警告使用者。 */
export async function storageQuota() {
  if (!navigator.storage?.estimate) return null;
  const { quota, usage } = await navigator.storage.estimate();
  return { quota, usage, free: quota - usage };
}

/**
 * 要求「持久化儲存」。
 *
 * 沒有這個的話，瀏覽器在空間吃緊時會默默把快取清掉 ——
 * 對一個要長期常駐當節點的分頁來說，那等於隨時可能要重下載幾 GB。
 */
export async function requestPersistence() {
  if (!navigator.storage?.persist) return { supported: false, granted: false };
  const already = await navigator.storage.persisted();
  if (already) return { supported: true, granted: true };
  return { supported: true, granted: await navigator.storage.persist() };
}
