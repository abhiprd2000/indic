# Tokenizer scan (Devanagari: BHTB, MGTB, Maithili web, Angika MT, HDTB, FLORES+ devtest bho/mag/mai/hin; measurement only)

| Tokenizer | Vocab | Type | Regex has \p{M} | Fert BHTB [95% CI] | Fert FLORES bho | Words split % | Cut at mark % | Tokens ending inside cluster % | Top cut characters | Fix (FLORES bho fert: orig, orig+cBPE K8000, scoped K8000) |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen/Qwen2.5-7B | 151665 | byte-level BPE with regex | False | 4.07 [4.03, 4.12] | 4.64 [4.61, 4.67] | 85.61 | 84.22 | 57.34 | U+093E 22.0%; U+0947 15.4%; U+094D 12.9% | 4.64, 2.45, 1.57 |
| Qwen/Qwen3-1.7B | 151669 | byte-level BPE with regex | False | 4.07 [4.03, 4.12] | 4.64 [4.61, 4.67] | 85.61 | 84.22 | 57.34 | U+093E 22.0%; U+0947 15.4%; U+094D 12.9% | 4.64, 2.45, 1.57 |
| meta-llama/Llama-3.1-8B | 128256 | byte-level BPE with regex | False | 2.50 [2.47, 2.53] | 2.75 [2.73, 2.77] | 85.53 | 84.22 | 51.29 | U+093E 22.0%; U+0947 15.4%; U+094D 12.9% | 2.75, 2.43, 1.55 |
| meta-llama/Llama-3.2-1B | 128256 | byte-level BPE with regex | False | 2.50 [2.47, 2.53] | 2.75 [2.73, 2.77] | 85.53 | 84.22 | 51.29 | U+093E 22.0%; U+0947 15.4%; U+094D 12.9% | 2.75, 2.43, 1.55 |
| google/gemma-3-4b-pt | 262145 | SentencePiece BPE (byte fallback) | False | 1.50 [1.48, 1.52] | 1.67 [1.65, 1.68] | 0.0 | 0.0 | 9.58 | none | not applicable |
| mistralai/Mistral-Nemo-Base-2407 | 131072 | byte-level BPE with regex | True | 2.05 [2.02, 2.07] | 2.26 [2.25, 2.28] | 6.79 | 0.0 | 29.25 | U+002C 32.1%; U+0964 22.6%; U+002E 13.6% | not needed |
| microsoft/phi-4 | 100352 | byte-level BPE with regex | False | 4.44 [4.39, 4.49] | 4.97 [4.93, 5.00] | 85.53 | 84.22 | 60.62 | U+093E 22.0%; U+0947 15.4%; U+094D 12.9% | 4.97, 2.45, 1.58 |
| allenai/Olmo-3-1025-7B | 100278 | byte-level BPE with regex | False | 4.44 [4.39, 4.49] | 4.97 [4.93, 5.00] | 85.53 | 84.22 | 60.62 | U+093E 22.0%; U+0947 15.4%; U+094D 12.9% | 4.97, 2.45, 1.58 |
| deepseek-ai/DeepSeek-V3 | 128815 | byte-level BPE with regex | True | 2.74 [2.71, 2.77] | 3.07 [3.05, 3.09] | 6.73 | 0.0 | 38.35 | U+002C 30.2%; U+0964 21.3%; U+002E 12.8% | not needed |
| HuggingFaceTB/SmolLM3-3B | 128256 | byte-level BPE with regex | False | 2.50 [2.47, 2.53] | 2.75 [2.73, 2.77] | 85.53 | 84.22 | 51.29 | U+093E 22.0%; U+0947 15.4%; U+094D 12.9% | 2.75, 2.43, 1.55 |
| xlm-roberta-base | 250002 | SentencePiece unigram | False | 1.63 [1.61, 1.65] | 1.76 [1.75, 1.78] | 0.0 | 0.0 | 11.4 | none | not applicable |
| google/mt5-base | 250100 | SentencePiece unigram | False | 2.03 [2.01, 2.05] | 2.15 [2.14, 2.16] | 0.0 | 0.0 | 19.25 | none | not applicable |
| ai4bharat/IndicBERTv2-MLM-only | 250000 | WordPiece | False | 1.31 [1.29, 1.32] | 1.47 [1.46, 1.48] | 6.63 | 0.0 | 8.61 | U+002C 30.8%; U+0964 21.8%; U+002E 13.1% | not applicable |
| tiktoken o200k_base | 200019 | tiktoken (BPE over bytes with regex) | True | 1.68 [1.66, 1.71] | 1.86 [1.84, 1.88] | 6.71 | 0.0 | 22.34 | U+002C 33.5%; U+0964 23.6%; U+002E 14.2% | not needed |

Fix status per tokenizer:

- Qwen/Qwen2.5-7B: applied
- Qwen/Qwen3-1.7B: applied
- meta-llama/Llama-3.1-8B: applied
- meta-llama/Llama-3.2-1B: applied
- google/gemma-3-4b-pt: not applicable: SentencePiece BPE (byte fallback) has no regex pre-tokenizer that cuts at marks; continued BPE with a scoped regex does not apply
- mistralai/Mistral-Nemo-Base-2407: not needed: pre-tokenizer does not cut at marks (0.0% of Devanagari words); regex letter class includes \p{M}: True
- microsoft/phi-4: applied
- allenai/Olmo-3-1025-7B: applied
- deepseek-ai/DeepSeek-V3: not needed: pre-tokenizer does not cut at marks (0.0% of Devanagari words); regex letter class includes \p{M}: True
- HuggingFaceTB/SmolLM3-3B: applied
- xlm-roberta-base: not applicable: SentencePiece unigram has no regex pre-tokenizer that cuts at marks; continued BPE with a scoped regex does not apply
- google/mt5-base: not applicable: SentencePiece unigram has no regex pre-tokenizer that cuts at marks; continued BPE with a scoped regex does not apply
- ai4bharat/IndicBERTv2-MLM-only: not applicable: WordPiece has no regex pre-tokenizer that cuts at marks; continued BPE with a scoped regex does not apply
- tiktoken o200k_base: not needed: pre-tokenizer does not cut at marks (0.0% of Devanagari words); regex letter class includes \p{M}: True

Notes: Qwen/Qwen2.5-7B: AutoTokenizer pre-tokenizer differs from repo tokenizer.json; Qwen/Qwen3-1.7B: AutoTokenizer pre-tokenizer differs from repo tokenizer.json; xlm-roberta-base: AutoTokenizer pre-tokenizer differs from repo tokenizer.json; google/mt5-base: no tokenizer.json to compare
