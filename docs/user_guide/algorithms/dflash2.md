# DFlash2

DFlash2 extends [DFlash](dflash.md) with local dynamic convolutions and a candidate selector. The convolution lets neighboring positions exchange local information while the draft block remains parallel. The selector then adds a low-rank, predecessor-conditioned transition score to the draft model's unary token scores.

## How It Works

### Local Convolution

Each draft layer wraps both attention and the MLP with a grouped causal convolution. A projection of the current hidden state produces two dynamic kernels: one for the sublayer input and one for its output. A learned base kernel is shared across positions, while channels in each group share the dynamic component.

Convolution never crosses an anchor-block boundary. This matters during training because Speculators flattens multiple anchored blocks into one token dimension.

### Candidate Selector

The unary draft logits first select `selector_top_k` candidates at each position. The selector scores each transition as

```text
unary(candidate)
  + dot(predecessor_codebook(previous) * hidden_projection(hidden),
        successor_codebook(candidate))
```

Inference starts with the anchor token and then conditions each position on the token selected at the preceding position. This preserves one parallel draft-model forward pass: only the small selector walks the candidate path.

## Training Objective

The public DFlash2 implementation specifies inference but does not publish its training objective. The current Speculators implementation is therefore an experimental split objective:

1. The configured DFlash-family compound loss trains the unary logits.
2. A separate hard cross-entropy term trains the selector over the unary top-K candidates while teacher-forcing the previous sequence token. If the target is outside that set, training replaces the weakest unary candidate with the target so the K-way loss remains defined. Validation loss uses the same replacement; serving and strict validation candidate/path metrics do not.

`--selector-loss-alpha` controls the second term. Both terms use the configured fixed exponential or D-PACE position weighting. The selector is never trained against a full-vocabulary corrected distribution, matching the candidate set it can rerank at serving time.

Validation reports unary candidate recall and target mass, teacher-forced selector accuracy, and the accepted prefix of a greedy self-conditioned path. The path begins at the verified anchor and feeds each selected token to the next edge score. Accepted-length metrics include one verified/bonus token and report both the realized selector path and the oracle unary-top-K path. `accept_len` remains the analytical TV-overlap estimate from unary logits; `eal` uses the selector path.

These training/validation path metrics use verifier logits and hidden states from the recorded sample prefixes. On sampled or off-policy data they remain proxies for serving acceptance; use offline evaluation for end-to-end acceptance. Candidate target mass is measured on the original unary top-K, without training-time target injection.

DFlash2 currently requires the full verifier vocabulary. Pruned draft vocabularies are rejected because current serving implementations select candidates before any draft-to-target vocabulary mapping.

This experimental implementation directly trains full-vocabulary predecessor and successor codebooks. That is an expanded free-codebook parameterization: the public checkpoint format contains materialized codebooks, but the public sources do not disclose whether those tensors were free parameters during training. In particular, this prototype should not be interpreted as reproducing the selector parameter count reported in the DFlash2 blog.

## Key Parameters

| Parameter               | Default | Description                              |
| ----------------------- | ------: | ---------------------------------------- |
| `--conv-kernel-size`    |       2 | Number of causal convolution taps        |
| `--conv-group-size`     |      16 | Hidden channels sharing a dynamic kernel |
| `--selector-rank`       |     256 | Rank of the transition factorization     |
| `--selector-top-k`      |      16 | Unary candidates reranked per position   |
| `--selector-loss-alpha` |     1.0 | Weight of the selector K-way CE term     |

All [DFlash](dflash.md) backbone parameters also apply. DFlash2 defaults to five draft layers, block size 8, `sample_from_anchor: False`, fixed exponential position weighting, and KL divergence loss. Set all shared knobs explicitly when comparing it with another algorithm.

To train your own, see `examples/train/dflash2_qwen3_8b_ultrachat_online_5k.sh`.

For the local Qwen3-4B Ascend comparison, use
`examples/train/qwen3_4b_bestarch/server.sh` and
`examples/train/qwen3_4b_bestarch/train_dflash2.sh`. It retains DSpark's training
budget: ten epochs, seven predicted tokens (DFlash2 block size 8), AdamW at
6e-4, cosine scheduling with 4% warmup, sequence length 3072, 512 anchors,
six training devices and the existing 700k prepared dataset. The unary loss
uses DFlash-style CE, with gamma 7 position weighting and the experimental
selector CE at weight 0.1. See that directory's README for the complete
comparison settings. Shared pruned
vocabulary caches are ignored for standalone DFlash2. Both the typed training
entrypoint and the legacy script accept standalone convolution/selector flags.

## Checkpoint loading and offline evaluation

Saved Speculators DFlash2 checkpoints load through `SpeculatorModel.from_pretrained`.
Missing trained convolution or selector tensors are errors. When resuming, explicit
convolution and selector shape options must match the checkpoint; omitted options
inherit its saved values.

External full-attention Qwen3-style DFlash2 checkpoints can be converted with:

```bash
speculators convert /path/to/external-dflash2 \
    --verifier /path/to/verifier --algorithm dflash2 --output-path ./converted-dflash2
```

The converter accepts raw codebooks and Embedding-style `.weight` keys, reads
draft options from either `dflash_config` or the top level, and preserves the
source's target-layer selection. It rejects source sliding-window masks, causal
full attention and non-default embedding/logit scaling, whose semantics the
local backbone cannot currently reproduce. Convert external checkpoints before
passing them to the training CLI.

`scripts/evaluate/dflash2_offline_eval.py` reuses the shared offline evaluator for
greedy or sampled verification. It evaluates the standalone selector without
requiring DSpark's Markov or MMuse's Correction heads. Run it with `--help` for
model, dataset and device options.

## Serving

Checkpoints use the public DFlash2 weight names with a Speculators configuration.
This repository supports training, checkpoint loading, conversion, and offline
evaluation. Deployment in vLLM requires a compatible DFlash2 runtime and
configuration loader; this adaptation has not validated vLLM serving compatibility.

## Research and Implementation Reference

The architecture follows Z Lab's [DFlash2 implementation](https://github.com/z-lab/dflash/blob/07ebd93db9f472af339b644bb70221ad8428328a/dflash/model.py) and the [DFlash2 technical blog](https://inco.ai/blog/dflash2/). The pinned source revision defines the inference architecture and checkpoint contract; the experimental training objective above is specific to this implementation.

## See Also

- [DFlash](dflash.md) -- The block-parallel draft backbone
- [DSpark](dspark.md) -- Adds a Markov head and confidence scheduling
- [Train a Speculator](../tutorials/train.md) -- End-to-end training workflow
