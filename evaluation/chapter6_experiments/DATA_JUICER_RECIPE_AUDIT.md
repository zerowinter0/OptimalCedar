# Data-Juicer recipe migration audit

Audit date: 2026-07-25

This audit covers the four-workload feasibility extension registered in
`DATA_JUICER_CANDIDATE_PROTOCOL.md`. The authoritative recipe source is
`datajuicer/data-juicer-hub@47fc345`; operator behavior is frozen to the local
`data-juicer` checkout at
`bb3d88aac183cc22b6f816262a812a9e5d5abb57`.

## Authoritative recipe digests

The files were read directly from the immutable GitHub revision. SHA-256:

| recipe | SHA-256 |
|---|---|
| `pile-hackernews-refine.yaml` | `bd70643116230c2a8226fe100ae12d886f667dc1e818481af3f9f32f8003a676` |
| `pile-pubmed-abstract-refine.yaml` | `5ca9fcb0205672411187d27853d0cac4d7417a9be9ac34bcdb30cf39927d8238` |
| `pile-freelaw-refine.yaml` | `ad6bfff434652710d7b4362ad196602c5747dd545ba916caadfadc3e9ea20fe7` |
| `pile-uspto-refine.yaml` | `f17772cb32433ed6459e1cc99072888cfdb32b5b929c2a298c9fa9e54c022d82` |

## Migration result

`evaluation/pipelines/pile_recipe_registry.py` matches every recipe's:

- mapper presence and order;
- filter presence and order;
- explicit operator argument;
- omitted argument through the frozen Data-Juicer default.

HackerNews correctly omits `clean_links_mapper`; the other three include it.
FreeLaw alone includes `stopwords_filter`. The final
`document_simhash_deduplicator` is the only omitted recipe operator. It is a
cross-record dataset operation and therefore outside Cedar's per-record
reorder/fusion search; this boundary is declared before measurement.

## Operator semantics

The migrated implementations were compared with the frozen Data-Juicer
classes for:

- alphanumeric ratio;
- average/maximum line length;
- character and word repetition;
- flagged-word and stopword ratio;
- FastText language score;
- SentencePiece/KenLM perplexity;
- special-character ratio;
- text length and tokenized word count;
- email/link cleaning and Unicode/punctuation/whitespace normalization.

The word-list assets use the same Data-Juicer URLs. Cached artifacts:

| asset | SHA-256 |
|---|---|
| `flagged_words.json` | `b9ce869ad4d9e92e979ce1dda36143452bb7c1733106bdab36ec08248f8d82df` |
| `stopwords.json` | `0867365c99ad8350f27cfd867c418d992c71df080bbe8eee8a99792e4fa11e88` |

The local `StopwordsFilter` tests only the configured lower bound, while the
frozen implementation also applies its default upper bound of `1.0`. Both
implementations cap the computed ratio at `1.0`, so these predicates are
equivalent for the registered FreeLaw arguments.

## Video self-evolution replacement candidate

The diverse-workload extension additionally audits
`refined_recipes/video/data-juicer-sandbox-self-evolution.yaml` from the same
Hub revision. Its SHA-256 is
`a593d79ca1cbe3acabb4432b96638a8a04f2efe0e7442a7449f233d9872e2b78`.
The Cedar migration preserves the five operators, order, and predicate
arguments exactly. The recipe's `mem_required` values are scheduler hints, not
predicate arguments; Cedar represents the same distinction by marking NSFW,
frame-text similarity, and aesthetics as CUDA-resource pipes. Parse,
video-root resolution, and output projection are fixed Cedar adapters and are
not counted as Hub filters. The executable verifier is
`evaluation/pipelines/video_self_evolution/validate_recipe.py`.
