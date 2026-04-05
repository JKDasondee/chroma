use std::collections::BinaryHeap;
use std::sync::Arc;

use async_trait::async_trait;
use chroma_blockstore::{BlockfileFlusher, BlockfileReader, BlockfileWriter};
use chroma_error::{ChromaError, ErrorCodes};
use chroma_types::{DirectoryBlock, SignedRoaringBitmap, SparsePostingBlock};
use std::iter;
use dashmap::DashMap;
use thiserror::Error;
use uuid::Uuid;

use crate::sparse::types::{decode_u32, encode_u32};

// ── Two-phase re-scoring ────────────────────────────────────────────

/// Batch-oriented rescorer for two-phase retrieval. MaxScore generates
/// an oversampled candidate set with approximate (quantized) scores;
/// the rescorer computes exact scores so the caller can pick the true
/// top-k.
#[async_trait]
pub trait SparseRescorer: Send + Sync {
    async fn rescore_batch(&self, doc_ids: &[u32], query: &[(u32, f32)]) -> Vec<f32>;
}

/// Re-score an oversampled candidate set and return the final top-k
/// by exact score.
pub async fn rescore_and_select(
    candidates: Vec<Score>,
    k: usize,
    query: &[(u32, f32)],
    rescorer: &dyn SparseRescorer,
) -> Vec<Score> {
    if candidates.is_empty() || k == 0 {
        return vec![];
    }

    let doc_ids: Vec<u32> = candidates.iter().map(|s| s.offset).collect();
    let exact_scores = rescorer.rescore_batch(&doc_ids, query).await;

    let mut heap: BinaryHeap<Score> = BinaryHeap::with_capacity(k);
    for (i, &score) in exact_scores.iter().enumerate() {
        if heap.len() < k || score > heap.peek().map(|s| s.score).unwrap_or(f32::MIN) {
            heap.push(Score {
                score,
                offset: doc_ids[i],
            });
            if heap.len() > k {
                heap.pop();
            }
        }
    }

    let mut results: Vec<Score> = heap.into_vec();
    results.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.offset.cmp(&b.offset)));
    results
}

const DEFAULT_BLOCK_SIZE: u32 = 1024;
const DIRECTORY_KEY: u32 = u32::MAX;

pub const SPARSE_POSTING_BLOCK_SIZE_BYTES: usize = 1024 * 1024;

#[derive(Debug, Error)]
pub enum BlockSparseError {
    #[error(transparent)]
    Blockfile(#[from] Box<dyn ChromaError>),
}

impl ChromaError for BlockSparseError {
    fn code(&self) -> ErrorCodes {
        match self {
            BlockSparseError::Blockfile(err) => err.code(),
        }
    }
}

// ── Score type ──────────────────────────────────────────────────────

/// A (score, offset) pair with reversed ordering so that `BinaryHeap`
/// acts as a min-heap: the *lowest* score sits at `peek()`, making it
/// cheap to maintain a top-k set.
#[derive(Debug, PartialEq)]
pub struct Score {
    pub score: f32,
    pub offset: u32,
}

impl Eq for Score {}

impl Ord for Score {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score
            .total_cmp(&other.score)
            .then(self.offset.cmp(&other.offset))
            .reverse()
    }
}

impl PartialOrd for Score {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

// ── BlockSparseFlusher ──────────────────────────────────────────────

pub struct BlockSparseFlusher {
    posting_flusher: BlockfileFlusher,
}

impl BlockSparseFlusher {
    pub async fn flush(self) -> Result<(), BlockSparseError> {
        self.posting_flusher
            .flush::<u32, SparsePostingBlock>()
            .await?;
        Ok(())
    }

    pub fn id(&self) -> Uuid {
        self.posting_flusher.id()
    }
}

// ── BlockSparseWriter ───────────────────────────────────────────────

#[derive(Clone)]
pub struct BlockSparseWriter<'me> {
    block_size: u32,
    delta: Arc<DashMap<u32, DashMap<u32, Option<f32>>>>,
    posting_writer: BlockfileWriter,
    old_reader: Option<BlockSparseReader<'me>>,
}

impl<'me> BlockSparseWriter<'me> {
    pub fn new(
        posting_writer: BlockfileWriter,
        old_reader: Option<BlockSparseReader<'me>>,
    ) -> Self {
        Self {
            block_size: DEFAULT_BLOCK_SIZE,
            delta: Default::default(),
            posting_writer,
            old_reader,
        }
    }

    pub fn with_block_size(mut self, block_size: u32) -> Self {
        self.block_size = block_size;
        self
    }

    pub async fn set(&self, offset: u32, sparse_vector: impl IntoIterator<Item = (u32, f32)>) {
        for (dimension_id, value) in sparse_vector {
            self.delta
                .entry(dimension_id)
                .or_default()
                .insert(offset, Some(value));
        }
    }

    pub async fn delete(&self, offset: u32, sparse_indices: impl IntoIterator<Item = u32>) {
        for dimension_id in sparse_indices {
            self.delta
                .entry(dimension_id)
                .or_default()
                .insert(offset, None);
        }
    }

    pub async fn commit(self) -> Result<BlockSparseFlusher, BlockSparseError> {
        let mut all_dim_ids: Vec<u32> = self.delta.iter().map(|e| *e.key()).collect();

        if let Some(ref reader) = self.old_reader {
            let old_dims = reader.get_all_dimension_ids().await?;
            all_dim_ids.extend(old_dims);
        }

        all_dim_ids.sort_unstable();
        all_dim_ids.dedup();

        let mut encoded_dims: Vec<(String, u32)> = all_dim_ids
            .into_iter()
            .map(|id| (encode_u32(id), id))
            .collect();
        encoded_dims.sort_unstable_by(|a, b| a.0.cmp(&b.0));

        for (encoded_dim, dimension_id) in &encoded_dims {
            let delta_updates = self.delta.remove(dimension_id);

            if delta_updates.is_none() {
                continue;
            }

            let (_, updates) = delta_updates.unwrap();

            let mut entries = std::collections::HashMap::new();
            let mut old_block_count = 0u32;
            if let Some(ref reader) = self.old_reader {
                let blocks = reader.get_posting_blocks(encoded_dim).await?;
                old_block_count = blocks.len() as u32;
                for block in blocks {
                    for (off, val) in block.offsets().iter().zip(block.values().iter()) {
                        entries.insert(*off, *val);
                    }
                }
            }

            for entry in updates.into_iter() {
                let (off, update) = entry;
                match update {
                    Some(val) => {
                        entries.insert(off, val);
                    }
                    None => {
                        entries.remove(&off);
                    }
                }
            }

            if entries.is_empty() {
                for seq in 0..old_block_count {
                    self.posting_writer
                        .delete::<_, SparsePostingBlock>(encoded_dim, seq)
                        .await?;
                }
                self.posting_writer
                    .delete::<_, SparsePostingBlock>(encoded_dim, DIRECTORY_KEY)
                    .await?;
                continue;
            }

            let mut sorted: Vec<(u32, f32)> = entries.into_iter().collect();
            sorted.sort_unstable_by_key(|(off, _)| *off);

            let mut dir_max_offsets = Vec::new();
            let mut dir_max_weights = Vec::new();

            let new_block_count = sorted.chunks(self.block_size as usize).len() as u32;
            for (seq, chunk) in sorted.chunks(self.block_size as usize).enumerate() {
                let block = SparsePostingBlock::from_sorted_entries(chunk)
                    .expect("chunk is non-empty and <= block_size");
                dir_max_offsets.push(block.max_offset);
                dir_max_weights.push(block.max_weight);
                self.posting_writer
                    .set(encoded_dim, seq as u32, block)
                    .await?;
            }

            for seq in new_block_count..old_block_count {
                self.posting_writer
                    .delete::<_, SparsePostingBlock>(encoded_dim, seq)
                    .await?;
            }

            let directory = DirectoryBlock::new(&dir_max_offsets, &dir_max_weights)
                .expect("directory: offsets/weights aligned by construction");
            self.posting_writer
                .set(encoded_dim, DIRECTORY_KEY, directory.into_block())
                .await?;
        }

        let flusher = self
            .posting_writer
            .commit::<u32, SparsePostingBlock>()
            .await?;

        Ok(BlockSparseFlusher {
            posting_flusher: flusher,
        })
    }
}

pub use super::cursor::PostingCursor;

// ── BlockSparseReader ───────────────────────────────────────────────

#[derive(Clone)]
pub struct BlockSparseReader<'me> {
    posting_reader: BlockfileReader<'me, u32, SparsePostingBlock>,
}

impl<'me> BlockSparseReader<'me> {
    pub fn new(posting_reader: BlockfileReader<'me, u32, SparsePostingBlock>) -> Self {
        Self { posting_reader }
    }

    pub fn posting_id(&self) -> Uuid {
        self.posting_reader.id()
    }

    pub fn posting_reader(&self) -> &BlockfileReader<'me, u32, SparsePostingBlock> {
        &self.posting_reader
    }

    pub async fn get_posting_blocks(
        &self,
        encoded_dim: &str,
    ) -> Result<Vec<SparsePostingBlock>, BlockSparseError> {
        let blocks: Vec<(u32, SparsePostingBlock)> =
            self.posting_reader.get_prefix(encoded_dim).await?.collect();
        Ok(blocks
            .into_iter()
            .filter(|(key, _)| *key != DIRECTORY_KEY)
            .map(|(_, b)| b)
            .collect())
    }

    pub async fn get_all_dimension_ids(&self) -> Result<Vec<u32>, BlockSparseError> {
        let all: Vec<(&str, u32, SparsePostingBlock)> =
            self.posting_reader.get_range(.., ..).await?.collect();

        let mut dims: Vec<u32> = all
            .iter()
            .filter_map(|(prefix, _, _)| decode_u32(prefix).ok())
            .collect();
        dims.sort_unstable();
        dims.dedup();
        Ok(dims)
    }

    /// Open a cursor for a dimension by loading all its posting blocks
    /// eagerly. Returns `None` if the dimension has no data.
    pub async fn open_cursor(
        &'me self,
        encoded_dim: &str,
    ) -> Result<Option<PostingCursor<'me>>, BlockSparseError> {
        let blocks = self.get_posting_blocks(encoded_dim).await?;
        if blocks.is_empty() {
            return Ok(None);
        }
        Ok(Some(PostingCursor::from_blocks(blocks)))
    }

    /// BlockMaxMaxScore query using the 3-batch I/O pipeline.
    ///
    /// 1. **Batch 1 — directories**: load directory blocks for every
    ///    query dimension in parallel, parse metadata.
    /// 2. **Batch 2 — essential data**: small dims (≤2 Arrow blocks)
    ///    get View cursors immediately; large dims get Lazy cursors
    ///    whose blocks are loaded and populated in bulk.
    /// 3. **Batch 3 — non-essential data**: after the threshold
    ///    stabilizes, load remaining blocks for non-essential terms,
    ///    pruning blocks that can't beat the threshold.
    pub async fn query(
        &'me self,
        query_vector: impl IntoIterator<Item = (u32, f32)>,
        k: u32,
        mask: SignedRoaringBitmap,
    ) -> Result<Vec<Score>, BlockSparseError> {
        if k == 0 {
            return Ok(vec![]);
        }

        let collected: Vec<(u32, f32)> = query_vector.into_iter().collect();
        let encoded_dims: Vec<String> = collected.iter().map(|(d, _)| encode_u32(*d)).collect();

        // ── Batch 1: load directory blocks ─────────────────────────
        let dir_keys: Vec<(String, u32)> = encoded_dims
            .iter()
            .map(|d| (d.clone(), DIRECTORY_KEY))
            .collect();
        self.posting_reader.load_data_for_keys(dir_keys).await;

        struct TermMeta {
            encoded_dim: String,
            dir_max_offsets: Vec<u32>,
            dir_max_weights: Vec<f32>,
            query_weight: f32,
            max_score: f32,
        }

        let mut metas: Vec<TermMeta> = Vec::new();
        for (idx, &(_, query_weight)) in collected.iter().enumerate() {
            let encoded_dim = encoded_dims[idx].clone();
            let dir_block = match self
                .posting_reader
                .get(&encoded_dim, DIRECTORY_KEY)
                .await?
            {
                Some(block) if block.is_directory() => DirectoryBlock::from_block(block).ok(),
                _ => None,
            };
            let Some(dir) = dir_block else { continue };
            let (dir_max_offsets, dir_max_weights) = dir.entries();
            if dir_max_offsets.is_empty() {
                continue;
            }
            let dim_max = dir_max_weights.iter().copied().fold(0.0f32, f32::max);
            let max_score = query_weight * dim_max;
            metas.push(TermMeta {
                encoded_dim,
                dir_max_offsets,
                dir_max_weights,
                query_weight,
                max_score,
            });
        }

        if metas.is_empty() {
            return Ok(vec![]);
        }

        // ── Build cursors ──────────────────────────────────────────
        // Small dimensions (≤2 Arrow blocks) use the eager View path;
        // large dimensions use Lazy cursors populated in Batch 2.
        let mut terms: Vec<TermState<'me>> = Vec::new();
        for meta in metas {
            let block_count =
                self.posting_reader.count_blocks_for_prefix(&meta.encoded_dim);

            if block_count <= 2 {
                self.posting_reader
                    .load_blocks_for_prefixes(iter::once(meta.encoded_dim.as_str()))
                    .await;
                let n = meta.dir_max_offsets.len();
                let raw_blocks: Vec<&[u8]> = (0..n)
                    .filter_map(|seq| {
                        self.posting_reader
                            .get_raw_from_cache(&meta.encoded_dim, seq as u32)
                    })
                    .collect();

                let mut cursor = if raw_blocks.len() == n {
                    PostingCursor::open(raw_blocks, meta.dir_max_offsets, meta.dir_max_weights)
                } else {
                    let blocks = self.get_posting_blocks(&meta.encoded_dim).await?;
                    if blocks.is_empty() {
                        continue;
                    }
                    PostingCursor::from_blocks(blocks)
                };
                cursor.advance(0, &mask);
                terms.push(TermState {
                    cursor,
                    encoded_dim: meta.encoded_dim,
                    query_weight: meta.query_weight,
                    max_score: meta.max_score,
                    window_score: meta.max_score,
                });
            } else {
                let cursor =
                    PostingCursor::open_lazy(meta.dir_max_offsets, meta.dir_max_weights);
                terms.push(TermState {
                    cursor,
                    encoded_dim: meta.encoded_dim,
                    query_weight: meta.query_weight,
                    max_score: meta.max_score,
                    window_score: meta.max_score,
                });
            }
        }

        if terms.is_empty() {
            return Ok(vec![]);
        }

        terms.sort_by(|a, b| a.max_score.total_cmp(&b.max_score));

        // ── Batch 2: load all blocks for essential terms ───────────
        // At threshold=MIN all terms are essential. Load their posting
        // blocks so the first windows can run without blocking.
        let essential_idx = 0usize;
        {
            let mut keys_to_load: Vec<(String, u32)> = Vec::new();
            for t in terms[essential_idx..].iter() {
                if t.cursor.is_lazy() {
                    for bk in 0..t.cursor.block_count() as u32 {
                        keys_to_load.push((t.encoded_dim.clone(), bk));
                    }
                }
            }
            if !keys_to_load.is_empty() {
                self.posting_reader.load_data_for_keys(keys_to_load).await;
                for t in terms[essential_idx..].iter_mut() {
                    if t.cursor.is_lazy() {
                        let dim = t.encoded_dim.clone();
                        t.cursor
                            .populate_all_from_cache(&self.posting_reader, &dim);
                    }
                }
            }
        }

        for t in terms.iter_mut() {
            if t.cursor.is_lazy() {
                t.cursor.advance(0, &mask);
            }
        }

        let mut non_essential_loaded = false;

        // ── Window loop ────────────────────────────────────────────
        let k_usize = k as usize;
        let mut threshold = f32::MIN;
        let mut heap: BinaryHeap<Score> = BinaryHeap::with_capacity(k_usize);

        const WINDOW_WIDTH: u32 = 4096;
        const BITMAP_WORDS: usize = (WINDOW_WIDTH as usize).div_ceil(64);
        let mut accum = vec![0.0f32; WINDOW_WIDTH as usize];
        let mut bitmap = [0u64; BITMAP_WORDS];
        let mut cand_docs: Vec<u32> = Vec::with_capacity(WINDOW_WIDTH as usize);
        let mut cand_scores: Vec<f32> = Vec::with_capacity(WINDOW_WIDTH as usize);

        let max_doc_id = terms
            .iter()
            .filter_map(|t| t.cursor.dir_max_offsets.last().copied())
            .max()
            .unwrap_or(0);

        let mut window_start = 0u32;

        while window_start <= max_doc_id {
            let window_end = (window_start + WINDOW_WIDTH - 1).min(max_doc_id);

            for t in terms.iter_mut() {
                t.window_score =
                    t.query_weight * t.cursor.window_upper_bound(window_start, window_end);
            }
            terms.sort_unstable_by(|a, b| a.window_score.total_cmp(&b.window_score));

            let mut essential_idx = terms.len();
            {
                let mut prefix = 0.0f32;
                for (i, t) in terms.iter().enumerate() {
                    prefix += t.window_score;
                    if prefix >= threshold {
                        essential_idx = i;
                        break;
                    }
                }
            }

            for term in terms[essential_idx..].iter_mut() {
                term.cursor.drain_essential(
                    window_start,
                    window_end,
                    term.query_weight,
                    &mut accum,
                    &mut bitmap,
                    &mask,
                );
            }

            cand_docs.clear();
            cand_scores.clear();
            for (word_idx, &word) in bitmap.iter().enumerate().take(BITMAP_WORDS) {
                let mut bits = word;
                while bits != 0 {
                    let bit = bits.trailing_zeros() as usize;
                    let idx = word_idx * 64 + bit;
                    cand_docs.push(window_start + idx as u32);
                    cand_scores.push(accum[idx]);
                    bits &= bits.wrapping_sub(1);
                }
            }

            if cand_docs.is_empty() {
                window_start = window_end.wrapping_add(1);
                if window_start == 0 {
                    break;
                }
                continue;
            }

            // ── Batch 3: lazy-load non-essential blocks (once) ────
            if !non_essential_loaded && essential_idx > 0 {
                non_essential_loaded = true;
                let mut ne_keys: Vec<(String, u32)> = Vec::new();
                for t in terms.iter() {
                    if !t.cursor.is_lazy() {
                        continue;
                    }
                    for bi in 0..t.cursor.block_count() {
                        if t.cursor.is_block_loaded(bi) {
                            continue;
                        }
                        if t.query_weight * t.cursor.dir_max_weights[bi] > threshold {
                            ne_keys.push((t.encoded_dim.clone(), bi as u32));
                        }
                    }
                }
                if !ne_keys.is_empty() {
                    self.posting_reader.load_data_for_keys(ne_keys).await;
                    for t in terms.iter_mut() {
                        if t.cursor.is_lazy() {
                            let dim = t.encoded_dim.clone();
                            t.cursor
                                .populate_all_from_cache(&self.posting_reader, &dim);
                        }
                    }
                }
            }

            if essential_idx > 0 {
                let mut remaining_budget: f32 =
                    terms[..essential_idx].iter().map(|t| t.window_score).sum();

                for i in (0..essential_idx).rev() {
                    if heap.len() >= k_usize && remaining_budget > 0.0 {
                        let cutoff = threshold - remaining_budget;
                        filter_competitive(&mut cand_docs, &mut cand_scores, cutoff);
                    }
                    if cand_docs.is_empty() {
                        break;
                    }

                    if terms[i].window_score == 0.0 {
                        continue;
                    }

                    let qw = terms[i].query_weight;
                    terms[i].cursor.score_candidates(
                        window_start,
                        window_end,
                        qw,
                        &cand_docs,
                        &mut cand_scores,
                    );

                    remaining_budget -= terms[i].window_score;
                }
            }

            for (ci, &doc) in cand_docs.iter().enumerate() {
                let score = cand_scores[ci];
                if score > threshold || heap.len() < k_usize {
                    heap.push(Score { score, offset: doc });
                    if heap.len() > k_usize {
                        heap.pop();
                    }
                    if heap.len() == k_usize {
                        threshold = heap.peek().map(|s| s.score).unwrap_or(f32::MIN);
                    }
                }
            }

            for (word_idx, word) in bitmap.iter_mut().enumerate().take(BITMAP_WORDS) {
                let mut bits = *word;
                while bits != 0 {
                    let bit = bits.trailing_zeros() as usize;
                    accum[word_idx * 64 + bit] = 0.0;
                    bits &= bits.wrapping_sub(1);
                }
                *word = 0;
            }

            window_start = window_end.wrapping_add(1);
            if window_start == 0 {
                break;
            }
        }

        let mut results: Vec<Score> = heap.into_vec();
        results.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.offset.cmp(&b.offset)));
        Ok(results)
    }
}

struct TermState<'a> {
    cursor: PostingCursor<'a>,
    encoded_dim: String,
    query_weight: f32,
    max_score: f32,
    window_score: f32,
}

// ── Budget pruning (scalar; SIMD added in PR #4) ────────────────────

/// Remove candidates whose score <= cutoff. Both parallel arrays are
/// compacted in-place.
fn filter_competitive(cand_docs: &mut Vec<u32>, cand_scores: &mut Vec<f32>, cutoff: f32) {
    debug_assert_eq!(cand_docs.len(), cand_scores.len());
    let n = cand_docs.len();
    let mut write = 0;
    for i in 0..n {
        if cand_scores[i] > cutoff {
            cand_docs[write] = cand_docs[i];
            cand_scores[write] = cand_scores[i];
            write += 1;
        }
    }
    cand_docs.truncate(write);
    cand_scores.truncate(write);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn score_min_heap_ordering() {
        let mut heap = BinaryHeap::new();
        heap.push(Score {
            score: 3.0,
            offset: 1,
        });
        heap.push(Score {
            score: 1.0,
            offset: 2,
        });
        heap.push(Score {
            score: 2.0,
            offset: 3,
        });
        assert_eq!(heap.peek().unwrap().score, 1.0);
        heap.pop();
        assert_eq!(heap.peek().unwrap().score, 2.0);
    }

    #[test]
    fn score_tiebreak_by_offset() {
        let a = Score {
            score: 1.0,
            offset: 10,
        };
        let b = Score {
            score: 1.0,
            offset: 20,
        };
        assert!(a > b); // reversed: higher offset = "lower" priority
    }

    #[test]
    fn filter_competitive_removes_below_cutoff() {
        let mut docs = vec![1, 2, 3, 4, 5];
        let mut scores = vec![0.1, 0.5, 0.2, 0.8, 0.3];
        filter_competitive(&mut docs, &mut scores, 0.25);
        assert_eq!(docs, vec![2, 4, 5]);
        assert_eq!(scores, vec![0.5, 0.8, 0.3]);
    }

    #[test]
    fn filter_competitive_empty() {
        let mut docs: Vec<u32> = vec![];
        let mut scores: Vec<f32> = vec![];
        filter_competitive(&mut docs, &mut scores, 0.0);
        assert!(docs.is_empty());
    }

    #[test]
    fn cursor_from_blocks_single() {
        let block = SparsePostingBlock::from_sorted_entries(&[(0, 0.5), (10, 0.9)]).unwrap();
        let cursor = PostingCursor::from_blocks(vec![block]);
        assert_eq!(cursor.block_count(), 1);
        assert_eq!(cursor.dimension_max(), 0.9);
    }

    #[test]
    fn cursor_advance_basic() {
        let block =
            SparsePostingBlock::from_sorted_entries(&[(5, 0.1), (10, 0.2), (15, 0.3), (20, 0.4)])
                .unwrap();
        let all = SignedRoaringBitmap::Exclude(Default::default());
        let mut cursor = PostingCursor::from_blocks(vec![block]);

        let r = cursor.advance(10, &all);
        assert_eq!(r, Some((10, 0.2)));

        let r = cursor.advance(16, &all);
        assert_eq!(r, Some((20, 0.4)));

        let r = cursor.advance(21, &all);
        assert_eq!(r, None);
    }

    #[test]
    fn cursor_get_value() {
        let block =
            SparsePostingBlock::from_sorted_entries(&[(5, 0.1), (10, 0.2), (15, 0.3)]).unwrap();
        let mut cursor = PostingCursor::from_blocks(vec![block]);

        assert_eq!(cursor.get_value(10), Some(0.2));
        assert_eq!(cursor.get_value(7), None);
        assert_eq!(cursor.get_value(99), None);
    }
}
