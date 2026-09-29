# Task 6 plan (fixed before any run; criteria below are not changed after results)

## Status
NOT RUN. Checked this session: no GPU (`nvidia-smi` missing), no PyTorch, 4 CPU cores, 15 GB RAM. This plan needs a GPU (cap 2.5 h). Waiting for a GPU environment or a decision on a CPU cut.

## Choices
- Tokenizer: variant d, K=8000, mix lowonly. Rule check from Task 5: d vs b mean fertility on the 5 languages 1.5002 vs 1.5004 (d not worse than b by >0.03). d is not perfectly clean at K=8000 (Bengali/Tamil change, 15 of 8000 new tokens) but b changes the same scripts and more at K=0, so d stays. Round trip: 0 failures. Load with `PreTrainedTokenizerFast(tokenizer_file=...)`.
- Model: Qwen/Qwen3-0.6B-Base, bf16. From its config.json: tie_word_embeddings=True, vocab_size=151936, hidden_size=1024, 28 layers. Tied: one matrix serves input and output; new rows are trained in that one matrix. Tokenizer has 151669 real ids; new ids 151669..159668. Resize to 159744 (multiple of 128); rows 151669.. are overwritten (old padding rows).
- Arms: A base model + base tokenizer. B extended tokenizer, new rows = mean of component base-token embeddings, no training. C = B trained on new rows only.
- Init: components come from the merge tree (`merges.json`) recursively down to original tokens (ids < 151669); never decode to text. Mean over the leaf tokens (with multiplicity).
- Training (C): freeze all but new rows (gradient mask on old rows; no weight decay). 30 MB, 1 epoch: English replay 15% (4.5 MB), Hindi 5% (1.5 MB), no Angika, remaining 80% (24 MB) over bho/mai/mag by size^0.3 (about 9.2 / 11.2 / 3.6 MB; Magahi is not repeated). English replay: HuggingFaceFW/fineweb-edu sample-10BT, first rows until 4.5 MB, seed 0, deduped against the English test text; if it cannot be fetched, stop. Seq 512, packed. Seed 0.
- LR: cosine, try {2e-4, 1e-3} for 300 steps each, pick by dev BPB (dev files of bho, mai, mag, hin, English replay-disjoint dev), then the full run.
- Compute rules: measure 20-step throughput first and print projected total time. Hard cap 2.5 h total GPU time. If projected steps < 500, cut data or sequence length, never the model. Checkpoints every 250 steps.

## Metrics
1. BPB on data/{bho,mai,mag,ang,hin,en}/test.txt, each sentence scored alone, prefixed with `<|endoftext|>`; BPB = total NLL / (ln2 x UTF-8 bytes). Arms A, B, C.
2. POS probe (Kumar et al. 2026 App. A.3): 2-layer MLP d->256->tags, GELU, LayerNorm; AdamW lr 2e-4, wd 0.01, cosine, 5% warmup, batch 32, 10 epochs, 3 seeds; last-layer states, first subword per whitespace word via offset mappings; train HDTB train, test BHTB and MGTB; UPOS tags. Text = UD word forms joined by spaces (so whitespace words = UD words). Report mean +- std. Kumar's Qwen3-1.7B numbers are not directly comparable (different model size).
3. Speed: 64 BHTB test sentences (seed 0), bf16, 5 warmups, 20 timed runs: tokens/sentence, ms/batch, bytes/sec, peak memory, A vs C.

## Success criteria (fixed)
- C BPB <= A BPB on bho, mai, mag.
- English BPB of C within +1% of A.
If not met: report plainly. One fallback allowed (double the steps if time allows); report both runs.

## Addendum (written before the first real run; hardware and unspecified details, criteria unchanged)
- Hardware: Kaggle kernel with 2x Tesla T4 (15 GB each). An RTX Pro 6000 request gave a CPU-only kernel. T4 has no native bf16, so compute is fp16 autocast over fp32 weights (GradScaler for training); this replaces "bf16". Check: arm A BPB on 200 bho test sentences in fp16 autocast vs bf16 (emulated) is reported. Speed numbers are fp16 on T4. One GPU is used.
- Trainable: a fp32 matrix of the 8000 new rows, shared by input embedding and the tied output head (old rows are a frozen buffer, so old rows cannot change). AdamW wd 0. 20 warmup steps, then cosine to 0.
- Batch: 4 sequences x 512 per micro-batch, 4 accumulation steps = 16 sequences (8192 tokens) per step. Units are shuffled, joined with `<|endoftext|>`, and cut into 512-token blocks.
- LR trials: 300 steps each from the same init; pick lower mean dev BPB over bho, mai, mag (first 300 dev sentences each; hin and English dev also logged). If a trial diverges (NaN), it loses.
- Compute rule: t = seconds per step from 20 steps. If (600 + full steps) x t > 2.5 h, cut the number of full steps (data) to fit; if fewer than 500 full steps would fit, halve seq length and re-measure. Never the model.
- BPB set: test sentences under 1500 UTF-8 bytes (same set for all arms; dropped count reported). Start token `<|endoftext|>`.
- POS probe: batch 32 sentences; features are last hidden states after the final norm, with `<|endoftext|>` prepended; accuracy over all test words; tags unseen in HDTB count as errors.
- English replay and dev come from HuggingFaceFW/fineweb-edu sample-10BT streamed in order (first docs); no test sentence appears in the replay text (checked in code).
- Memory (found in a smoke run: 4x512 micro-batches ran out of memory on a 15 GB T4): micro-batch 2 sequences x 512 with 8 accumulation steps; the step is still 16 sequences (8192 tokens). No criteria change.
