/**
 * Stable counting sort of `n` items by small integer keys (layer × pages + page), in O(n + buckets) with no
 * allocation: `counts` needs `buckets + 1` slots and `order` `n`. After the call `order` lists item indices sorted
 * by key, items with equal keys in push order, and `counts[k]..counts[k + 1]` is bucket k's span of `order`.
 */
export function countingSort(keys: Uint16Array, n: number, buckets: number, counts: Uint32Array, order: Uint32Array) {
  counts.fill(0, 0, buckets + 1);
  for (let i = 0; i < n; i++) counts[keys[i] + 1]++;
  for (let k = 0; k < buckets; k++) counts[k + 1] += counts[k];
  // Scatter with a running cursor per bucket; `counts[k]` ends at bucket k's end, so rebuild the starts afterwards
  for (let i = 0; i < n; i++) order[counts[keys[i]]++] = i;
  for (let k = buckets; k > 0; k--) counts[k] = counts[k - 1];
  counts[0] = 0;
}

/**
 * Consecutive equal keys in sorted order become one draw run. Writes run keys, first positions in `order` and
 * lengths; returns the run count.
 */
export function buildRuns(
  counts: Uint32Array,
  buckets: number,
  runKey: Uint16Array,
  runStart: Uint32Array,
  runCount: Uint32Array
): number {
  let runs = 0;
  for (let k = 0; k < buckets; k++) {
    const len = counts[k + 1] - counts[k];
    if (len === 0) continue;
    runKey[runs] = k;
    runStart[runs] = counts[k];
    runCount[runs] = len;
    runs++;
  }
  return runs;
}
