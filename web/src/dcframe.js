/**
 * Chunk 層：把一則 frame 切成 16 KiB 送出、在對面重組（docs/01-architecture.md §4.5.1、§4.5.6）。
 *
 * 這一層刻意**不 import** `wire.js`，也完全不碰 `RTCPeerConnection`：
 * 它搬的是不透明的 ArrayBuffer。這樣 frame 層不必知道 DataChannel 存在，
 * 而這一層可以用假 channel 單元測試 —— 真正的 DataChannel 在容器裡要兩個
 * browser context 才跑得起來，把它綁進單元測試等於讓這層永遠沒有回歸測試。
 *
 * 這個模組存在的唯一理由，是 §4.4 實測到的那個失敗模式：
 * 單次 send() 一則超過協商上限的訊息**不會丟例外**，正常返回之後才非同步
 * 觸發 `error { errorDetail: 'data-channel-failure' }` 並關閉 channel，
 * 0 位元組送達。看起來像對方斷線，實際上是自己送壞的 ——
 * 所以「永遠不要送出單一則大於上限的訊息」必須由程式保證，不能靠呼叫端自律。
 */

/** 分塊大小。不是取自 maxMessageSize，理由見下方 SIZING 說明。 */
export const CHUNK_SIZE = 16384;

/** 子標頭：messageId u32 + chunkIndex u16 + chunkCount u16（§4.5.6）。 */
export const CHUNK_HEADER_SIZE = 8;

/**
 * SIZING：每個 chunk 的**總長**（含子標頭）不得超過 CHUNK_SIZE，
 * 所以酬載上限是 16384 − 8 = 16376，不是 16384。
 *
 * 這個減法漏掉的話，每個 chunk 會變成 16392 位元組。它不會在本機炸開 ——
 * 16392 遠小於協商出來的 262144，測試會全綠 —— 但對上 SDP 沒宣告
 * `a=max-message-size` 的對端（RFC 8841 的預設是 65536 ... 而某些實作是
 * 16384）就會踩到上面那個非同步殺 channel 的路徑。所以邊界值有測試看著。
 */
const MAX_CHUNK_PAYLOAD = CHUNK_SIZE - CHUNK_HEADER_SIZE;

/** chunkCount 是 u16，所以一則 frame 最多 65535 個 chunk。 */
const MAX_CHUNKS = 0xffff;

/**
 * 背壓門檻。這兩個數字是為了**限制記憶體**，不是為了保住 channel。
 *
 * 先前根據 dcSCTP 原始碼推測「送出佇列堆到 2,000,000 位元組會殺掉 channel」，
 * 那個推測**沒有重現**：§4.4 實測用 16 KiB 分塊完全不做背壓，
 * bufferedAmount 堆到 3,145,728（3 MiB）沒事，繼續堆到 16,777,216（16 MiB）
 * 也沒事 —— 到 16 MiB 時 send() 丟出的是**可以 catch 的** OperationError，
 * channel 仍然是 open。
 *
 * 把這個 WHY 記在這裡，是因為下一個人如果以為門檻是在「避免 channel 被殺」，
 * 他會發現 channel 根本不會被殺，然後很合理地把整段背壓拿掉 ——
 * 於是一則 3 MiB 的 logits 會在 JS heap 裡再複製一份排隊，手機節點直接 OOM。
 * 真正不能拿掉的是 CHUNK_SIZE 的分塊；背壓拿掉只是變耗記憶體。
 */
const LOW_WATER = 262144;
const HIGH_WATER = 524288;

/** 同時重組中的訊息數上限。防的是「送一百萬個 chunk 0」這種 OOM 攻擊。 */
const MAX_IN_FLIGHT = 8;

/**
 * 記住「最近完成的 messageId」幾則，用來擋重播。
 *
 * 沒有這張表的時候，重複偵測只蓋得到**重組中**的訊息：一旦最後一塊到了、
 * entry 被 delete，同一則訊息再餵一次就變成全新的訊息。實測
 * `handle(mk(1,0,1)); handle(mk(1,0,1));` 會讓 onMessage **觸發兩次**、
 * 完全不報錯 —— 下游把同一個 frame 算兩遍，而模組自己的錯誤訊息卻說
 * 「DataChannel 是 reliable + ordered，不該出現重送」。多 chunk 的版本更糟：
 * 重播 chunk 0 會開一個永遠補不滿的新 entry，白佔一個 in-flight 名額到 channel 重建為止。
 *
 * 刻意是**有界**的環狀緩衝：記住全部 messageId 才是真的無上限成長（那正是
 * maxInFlight 在防的事）。代價是超過 64 則之前的重播抓不到 —— 這一層要抓的是
 * 程式錯誤與壞掉的對端，不是惡意重放；真正的防重放要序號窗口 + 認證（M6）。
 */
const COMPLETED_MEMORY = 64;

/**
 * 建立分塊送出端。
 *
 * @param {RTCDataChannel|object} channel 只用到 send / bufferedAmount /
 *   bufferedAmountLowThreshold / addEventListener，所以假 channel 也吃得下。
 * @param {object} [opts] highWater、lowWater、firstMessageId（測 u32 繞回用）。
 */
export function createSender(channel, opts = {}) {
  const highWater = opts.highWater ?? HIGH_WATER;
  const lowWater = opts.lowWater ?? LOW_WATER;
  // 先驗「是不是有限數」，再比大小。只留下面那個 lowWater >= highWater 的比較，
  // NaN 會整組靜默溜過去：NaN 的任何比較都是 false，所以建構成功，
  // 之後 `bufferedAmount > highWater` 也永遠是 false —— 背壓從此不存在。
  // 實測 createSender(ch, { highWater: NaN }) 送 3 MiB：backpressureWaits 是 0、
  // bufferedAmount 一路堆到 10,000,000，沒有任何錯誤，只是「看起來比較快」。
  // 所以門檻一定要用 Number.isFinite 驗，不能靠比較。
  for (const [name, value] of [['highWater', highWater], ['lowWater', lowWater]]) {
    if (!Number.isFinite(value) || value < 0) {
      throw new Error(
        `背壓門檻 ${name} 必須是非負的有限數，收到的是 ${value}。` +
        'NaN 或 Infinity 不會讓比較丟錯，只會讓背壓靜默失效、記憶體無上限地堆。',
      );
    }
  }
  if (lowWater >= highWater) {
    throw new Error(
      `背壓門檻設反了：lowWater(${lowWater}) 必須小於 highWater(${highWater})，` +
      '否則等到的 bufferedamountlow 仍然在高水位之上，會立刻再等一次。',
    );
  }

  // bufferedAmountLowThreshold 只設一次。每次送前重設會讓「已經在門檻之下」
  // 的狀態不再產生新的邊緣事件 —— 這個事件是邊緣觸發的，不是準位觸發。
  channel.bufferedAmountLowThreshold = lowWater;

  let nextMessageId = (opts.firstMessageId ?? 0) >>> 0;
  let messagesSent = 0;
  let chunksSent = 0;
  let bytesSent = 0;
  let backpressureWaits = 0;
  let maxBufferedAmount = 0;
  // 串成一條鏈：兩則訊息同時 send() 時不要交錯搶背壓額度。
  // 接收端本來就支援交錯（見 createReceiver），這裡串行化只是讓
  // 「誰先把 chunk 交給 channel」可預期，方便量測與除錯。
  let tail = Promise.resolve();

  /**
   * 等一次 bufferedamountlow。**不是**輪詢。
   *
   * 早期版本這裡是 `while (channel.bufferedAmount > highWater) await sleep(0)`，
   * 在 3 MiB 的 logits 上會空轉數十萬次、把事件迴圈佔滿，
   * 結果背壓本身變成延遲來源。所以一律用事件。
   */
  function waitForDrain() {
    return new Promise((resolve, reject) => {
      const onLow = () => {
        cleanup();
        resolve();
      };
      const onClose = () => {
        cleanup();
        reject(new Error(
          'DataChannel 在等待 bufferedamountlow 時關閉了，這則訊息只送出了一部分。' +
          '請重新建立 channel 後整則重送 —— chunk 層沒有續傳，半則訊息在對面是拼不回來的。',
        ));
      };
      function cleanup() {
        channel.removeEventListener?.('bufferedamountlow', onLow);
        channel.removeEventListener?.('close', onClose);
        channel.removeEventListener?.('error', onClose);
      }
      channel.addEventListener('bufferedamountlow', onLow, { once: true });
      channel.addEventListener?.('close', onClose, { once: true });
      channel.addEventListener?.('error', onClose, { once: true });
      // 註冊完再確認一次：事件是邊緣觸發的，如果在「讀到高於水位」到
      // 「掛上監聽器」之間剛好排空，那個邊緣就錯過了，會永遠等下去。
      // 這是一次性的補救讀取，不是迴圈。
      if (channel.bufferedAmount <= lowWater) onLow();
    });
  }

  async function sendOne(arrayBuffer) {
    const bytes = asBytes(arrayBuffer, 'send() 的參數');
    // 長度 0 也要送一個純子標頭的 chunk：接收端靠 chunkCount 判斷完成，
    // 完全不送的話對面會永遠等不到這則訊息，而不是收到一則空訊息。
    const chunkCount = Math.max(1, Math.ceil(bytes.length / MAX_CHUNK_PAYLOAD));
    if (chunkCount > MAX_CHUNKS) {
      throw new Error(
        `訊息 ${bytes.length} 位元組要切成 ${chunkCount} 個 chunk，` +
        `超過 chunkCount u16 的上限 ${MAX_CHUNKS}（約 ${MAX_CHUNKS * MAX_CHUNK_PAYLOAD} 位元組）。` +
        '請在 frame 層先把它拆成多則訊息。',
      );
    }

    const messageId = nextMessageId;
    // 明確用 >>> 0 繞回：u32 寫進 DataView 時會自動截斷，但計數器本身
    // 在 JS 裡會一路長成 2^53 的浮點數，之後與收端的 u32 對不上。
    nextMessageId = (nextMessageId + 1) >>> 0;

    for (let i = 0; i < chunkCount; i++) {
      if (channel.bufferedAmount > highWater) {
        backpressureWaits++;
        await waitForDrain();
      }
      const start = i * MAX_CHUNK_PAYLOAD;
      const end = Math.min(start + MAX_CHUNK_PAYLOAD, bytes.length);
      const chunk = new Uint8Array(CHUNK_HEADER_SIZE + (end - start));
      const view = new DataView(chunk.buffer);
      view.setUint32(0, messageId, true);
      view.setUint16(4, i, true);
      view.setUint16(6, chunkCount, true);
      chunk.set(bytes.subarray(start, end), CHUNK_HEADER_SIZE);
      // 傳 chunk.buffer 而不是 chunk：這個 Uint8Array 是剛配出來、剛好塞滿的，
      // 所以兩者長度相同。但若哪天改成共用一塊大 buffer 的 subarray，
      // .buffer 會是整塊 —— 那就會送出超大訊息，正是本模組要防的事。
      channel.send(chunk.buffer);
      chunksSent++;
      bytesSent += chunk.byteLength;
      // 讀一次就好。真實的 bufferedAmount 每次讀都可能不同（SCTP 在背景排空），
      // 讀兩次做比較與賦值會記到兩個不同的值，量出來的峰值反而偏低。
      const buffered = channel.bufferedAmount;
      if (buffered > maxBufferedAmount) maxBufferedAmount = buffered;
    }
    messagesSent++;
  }

  return {
    /** 全部 chunk 都交給 channel 之後才 resolve（不代表對面收到了）。 */
    send(arrayBuffer) {
      const p = tail.then(() => sendOne(arrayBuffer));
      // tail 要吞掉例外，否則一次失敗會讓之後所有 send() 都被同一個錯誤拒絕。
      tail = p.catch(() => {});
      return p;
    },
    stats() {
      return {
        messagesSent,
        chunksSent,
        bytesSent,
        backpressureWaits,
        maxBufferedAmount,
        nextMessageId,
      };
    },
  };
}

/**
 * 建立重組端。
 *
 * @param {(buffer: ArrayBuffer) => void} onMessage 收到完整訊息時呼叫。
 * @param {object} [opts] maxInFlight：同時重組中的訊息數上限。
 */
export function createReceiver(onMessage, opts = {}) {
  if (typeof onMessage !== 'function') {
    throw new Error('createReceiver(onMessage) 需要一個 callback 來接收重組完成的 ArrayBuffer。');
  }
  const maxInFlight = opts.maxInFlight ?? MAX_IN_FLIGHT;
  // 同上：NaN 的比較永遠是 false，所以 `partials.size >= maxInFlight` 這道上限
  // 會靜默失效，「送一百萬個 chunk 0」就真的能把記憶體堆爆 —— 而那正是這個
  // 上限唯一的存在理由。用 Number.isInteger 驗，不用比較。
  if (!Number.isInteger(maxInFlight) || maxInFlight < 1) {
    throw new Error(
      `maxInFlight 必須是 ≥ 1 的整數，收到的是 ${maxInFlight}。` +
      'NaN / Infinity / 小數都會讓同時重組中的訊息數上限靜默失效。',
    );
  }
  /** messageId -> { chunkCount, received, parts: Array<Uint8Array> } */
  const partials = new Map();
  /**
   * 最近完成的 messageId（環狀，見 COMPLETED_MEMORY）。
   * 填 -1 是因為 messageId 是 u32，-1 永遠不會是合法值，不必另外記「有沒有填過」。
   * 用普通 Array 而不是 Int32Array：messageId 0xffffffff 存進 Int32Array 會變成 -1，
   * 剛好撞上這個哨兵值 —— 繞回測試用的正是那個 id。
   */
  const completed = new Array(COMPLETED_MEMORY).fill(-1);
  let completedCursor = 0;

  /**
   * 餵進一個 chunk。可以直接餵 MessageEvent，也可以餵 ArrayBuffer。
   *
   * 所有錯誤都用 throw，不是靜默丟棄：少一塊 chunk 的張量不會報錯，
   * 只會讓下游算出「看起來合理但慢慢偏掉」的結果 —— 那種錯最難查。
   */
  function handle(eventOrBuffer) {
    const data = eventOrBuffer && eventOrBuffer.data !== undefined
      ? eventOrBuffer.data
      : eventOrBuffer;
    // Blob 才是「忘了設 binaryType」的真正症狀（§4.4）：binaryType 的預設值是
    // 'blob'，忘了設的那一端 onmessage 拿到的就是 Blob，不是字串。
    // 這兩個提示原本掛反了 —— 可以操作的那一句掛在字串分支，而字串跟 binaryType
    // 一點關係也沒有（那是對端真的送了文字），於是唯一需要提示的情況拿到的是
    // 「必須是 ArrayBuffer」這種查不出所以然的通用訊息。不要再換回去。
    if (isBlob(data)) {
      throw new Error(
        '收到 Blob。兩端都必須明確設 binaryType = \'arraybuffer\'（§4.4）—— ' +
        '包含 ondatachannel 收到的那一個，各家引擎的預設值歷史上並不一致。' +
        'Blob 只能非同步讀出位元組，chunk 層不接受，也不該偷偷幫你 await。',
      );
    }
    if (typeof data === 'string') {
      throw new Error(
        '收到字串訊息。chunk 層只吃二進位 chunk —— 對端把文字（例如 JSON 控制訊息）' +
        '送到了同一條 DataChannel 上，或根本不是用這個版本的 dcframe 送的。',
      );
    }
    const bytes = asBytes(data, '收到的 chunk');

    if (bytes.byteLength < CHUNK_HEADER_SIZE) {
      throw new Error(
        `chunk 只有 ${bytes.byteLength} 位元組，連 ${CHUNK_HEADER_SIZE} 位元組的子標頭都放不下。` +
        '對端不是用這個版本的 dcframe 送的，或訊息在途中被截斷了。',
      );
    }
    // 另一個方向的越界同樣要擋：單一個 200000 位元組的 chunk 原本會被照單全收，
    // 重組出一則遠大於 CHUNK_SIZE 的訊息 —— 但沒有任何合法送出端能產生它，
    // 收到就代表對端壞了。收下來只會把「誰送壞了」這件事拖到更下游才爆。
    if (bytes.byteLength > CHUNK_SIZE) {
      throw new Error(
        `chunk 總長 ${bytes.byteLength} 位元組超過 CHUNK_SIZE ${CHUNK_SIZE}（§4.5.6）。` +
        '合法的送出端每塊最多送 8 + 16376 位元組，不可能產生這種 chunk。',
      );
    }
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const messageId = view.getUint32(0, true);
    const chunkIndex = view.getUint16(4, true);
    const chunkCount = view.getUint16(6, true);

    if (chunkCount === 0) {
      throw new Error(
        `訊息 ${messageId} 的 chunkCount 是 0。最小的一則訊息也有 1 個 chunk（長度 0 也算），` +
        '這個值不可能合法。',
      );
    }
    if (chunkIndex >= chunkCount) {
      throw new Error(
        `訊息 ${messageId} 的 chunkIndex ${chunkIndex} 超出 chunkCount ${chunkCount}。` +
        '兩者都是 u16，合法範圍是 0 ≤ chunkIndex < chunkCount。',
      );
    }

    // 酬載長度也要驗，而且要在建 entry 之前驗（無狀態的檢查先做，畸形的 chunk
    // 不該佔掉一個 in-flight 名額）。
    //
    // 少了這一段的實際症狀：`handle(chunk(id=1, idx=0, cnt=2, 100 位元組))` 之後
    // `handle(chunk(id=1, idx=1, cnt=2, 4 位元組))`，onMessage 收到一則 **104 位元組**
    // 的訊息、inFlight 歸零、全程無錯 —— 但 cnt=2 的合法送出端一定送了 16376+N。
    // 被截斷的那一塊就這樣靜默變成「短一截但長得很合理」的張量，正是本模組開頭
    // 說要防的那種錯：下游不會炸，只會慢慢偏掉。
    const payloadLen = bytes.byteLength - CHUNK_HEADER_SIZE;
    const isLast = chunkIndex === chunkCount - 1;
    if (!isLast && payloadLen !== MAX_CHUNK_PAYLOAD) {
      throw new Error(
        `訊息 ${messageId} 的 chunk ${chunkIndex}（共 ${chunkCount} 塊）帶了 ${payloadLen} ` +
        `位元組的酬載，但非最後一塊必須剛好是 ${MAX_CHUNK_PAYLOAD} 位元組。` +
        '這塊被截斷了，或對端的分塊算術和這裡不一致。',
      );
    }
    // 最後一塊只驗下界：上界由前面那道「總長不得超過 CHUNK_SIZE」蓋掉了，
    // 在這裡再寫一次 payloadLen > MAX_CHUNK_PAYLOAD 會是永遠為假的死條件，
    // 沒有任何輸入能讓測試釘住它。
    // 長度 0 只有「整則訊息就是空的」時才合法（chunkCount === 1）；
    // 多塊訊息的最後一塊是 0，代表送出端多切了一塊，同樣是算術不一致。
    const minLast = chunkCount === 1 ? 0 : 1;
    if (isLast && payloadLen < minLast) {
      throw new Error(
        `訊息 ${messageId} 的最後一塊帶了 ${payloadLen} 位元組的酬載，` +
        `合法範圍是 ${minLast}..${MAX_CHUNK_PAYLOAD}。`,
      );
    }

    // 重播一則**已經交付**的訊息要丟錯，不是當成新訊息重收一遍。見 COMPLETED_MEMORY。
    if (isCompleted(messageId)) {
      throw new Error(
        `訊息 ${messageId} 已經重組完成並交付過了，這個 chunk ${chunkIndex} 是重播。` +
        `DataChannel 是 reliable + ordered，不該出現重送（§4.4）；` +
        `收下它會讓同一則 frame 被下游算兩次。` +
        `（只記得最近 ${COMPLETED_MEMORY} 則，更早的重播這一層抓不到。）`,
      );
    }

    let entry = partials.get(messageId);
    if (!entry) {
      if (partials.size >= maxInFlight) {
        throw new Error(
          `同時重組中的訊息已達上限 ${maxInFlight} 則（正在等：${[...partials.keys()].join(', ')}），` +
          `訊息 ${messageId} 被拒絕。這通常代表對端在亂送 chunk 0 —— ` +
          '請重建 channel；若是正常流量需要更多併發，調整 createReceiver 的 maxInFlight。',
        );
      }
      entry = { chunkCount, received: 0, parts: new Array(chunkCount) };
      partials.set(messageId, entry);
    } else if (entry.chunkCount !== chunkCount) {
      throw new Error(
        `訊息 ${messageId} 的 chunkCount 前後不一致：先前是 ${entry.chunkCount}，` +
        `這個 chunk 說是 ${chunkCount}。同一個 messageId 被重用或資料損毀了。`,
      );
    } else if (entry.parts[chunkIndex] !== undefined) {
      throw new Error(
        `訊息 ${messageId} 的 chunk ${chunkIndex} 重複了。DataChannel 是 reliable + ordered，` +
        '不該出現重送 —— 不要設 maxRetransmits / maxPacketLifeTime（§4.4）。',
      );
    }

    // 一定要複製：真實 DataChannel 每次給的是新 buffer，但假 channel、
    // 或未來某個共用 buffer 的最佳化會把同一塊記憶體覆寫掉，
    // 重組出來就是一堆最後一個 chunk 的內容。
    entry.parts[chunkIndex] = bytes.slice(CHUNK_HEADER_SIZE);
    entry.received++;
    if (entry.received < entry.chunkCount) return;

    partials.delete(messageId);
    rememberCompleted(messageId);
    let total = 0;
    for (const part of entry.parts) total += part.byteLength;
    const out = new Uint8Array(total);
    let off = 0;
    for (const part of entry.parts) {
      out.set(part, off);
      off += part.byteLength;
    }
    onMessage(out.buffer);
  }

  function isCompleted(messageId) {
    return completed.indexOf(messageId) >= 0;
  }

  function rememberCompleted(messageId) {
    completed[completedCursor] = messageId;
    completedCursor = (completedCursor + 1) % COMPLETED_MEMORY;
  }

  return {
    handle,
    stats() {
      return { inFlight: partials.size, maxInFlight, completedMemory: COMPLETED_MEMORY };
    },
  };
}

/**
 * 是不是 Blob。跨 realm（iframe / worker / 不同 window）的 Blob 不會通過
 * instanceof，所以 instanceof 失敗之後還要再看一次形狀 —— 這個分支要給的是
 * 「你忘了設 binaryType」那句提示，判斷不準就等於沒有提示。
 */
function isBlob(v) {
  if (typeof Blob !== 'undefined' && v instanceof Blob) return true;
  return !!v && typeof v === 'object'
    && typeof v.arrayBuffer === 'function'
    && typeof v.size === 'number'
    && typeof v.type === 'string';
}

/** ArrayBuffer / TypedArray / DataView 一律看成位元組視圖，不複製。 */
function asBytes(data, what) {
  if (data instanceof ArrayBuffer) return new Uint8Array(data);
  if (ArrayBuffer.isView(data)) return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  throw new Error(`${what} 必須是 ArrayBuffer 或 TypedArray，收到的是 ${typeName(data)}。`);
}

function typeName(v) {
  if (v === null) return 'null';
  return typeof v === 'object' ? (v.constructor?.name ?? 'object') : typeof v;
}
