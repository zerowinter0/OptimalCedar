# Target pipelines

The five production entrypoints are explicit Cedar Features. Read each
workload's _compose() for operators, tags and dependencies.

| Workload | Entry | Output | Dependency contract |
| --- | --- | --- | --- |
| SimCLRv2 | simclr/cedar_dataset.py | Native single-view tensor batch | Original Cedar Feature: crop before flip; float before normalize; reader/batcher fixed |
| DINO | dino/cedar_dataset.py | 2 global 224px + 8 local 96px views | Each view: crop before flip; all PIL transforms before tensor; tensor before normalize |
| SwAV | swav/cedar_dataset.py | 2 global 224px + 6 local 96px views | Same dependency policy as DINO |
| CLIP | clip/cedar_dataset.py | Image tensor and token tensors | resize before crop; crop and float before normalize; tokenize before text tensor |
| BLIP | blip/cedar_dataset.py | Image tensor and cleaned caption | crop before flip; flip and RandAugment before tensor; tensor before normalize; clean before truncate |

DINO/SwAV jitter, grayscale and blur (and DINO solarization) are separate
MapperPipes. Each transforms the current value of its own view field.
Views remain independent and retain original output positions.
BLIP RandAugment retains its two-operation random policy as one operator.
Its NumPy result is converted back to PIL without changing pixel values, so
crop and flip can run after it. This adapter is shared by every optimizer.

## Reordering semantics

These are Cedar-style relaxed augmentation workloads. The default order
preserves the pinned recipes and parameters. Permitted reordered augmentation
sequences may change pixel values and augmentation distributions; they do not
claim exact equivalence to the pinned training preprocessing or unchanged
training accuracy. Blur before crop acts at a different effective scale;
BLIP RandAugment before crop changes the image seen by that crop.
CLIP float-before-resize can change interpolation rounding. Resize-before-
center-crop remains mandatory to preserve its geometry.

Conversion/type boundaries, output dimensions, view counts, caption word
boundaries and tokenizer outputs are preserved. Normalization remains terminal
within each new image/view path. All optimizers must use the same Features
and constraints in comparisons.

Random transforms use upstream random generators, like native SimCLRv2;
the previous per-stage RNG save/restore and thread lock are absent.
Different plans need not consume identical random draws. Record order and
worker seeding must be controlled by the experiment protocol.

## Inputs and entrypoints

cedar_dataset.get_dataset dispatches by spec.kwargs["workload"]:
simclr (also simclrv2), dino, swav, clip, blip.
Individual workload modules also expose get_dataset(spec).

SimCLRv2 copies the native module, including dataset construction and optimizer
options. The only relocation adjustment anchors its dataset path to the
original module. It reads the original Imagenette directory, not the former
two-view JSONL adapter. Its Feature and transforms are unchanged.

The other four use dataset_path=<manifest.jsonl> and optional image_root.
Records contain an image path; CLIP/BLIP also require caption. CLIP accepts
tokenizer_path (default openai/clip-vit-base-patch32, locally cached).
runtime.py shares dataset/optimizer setup; fields.py contains record adapters.
Batch sizes and source recipe parameters have not been reduced.

catalog.py, core.py, and legacy_dataset.py are retained only for strict
reference validation/backward compatibility. They are not production graphs.
build_workload() is that legacy reference API, not the new dependency graph.
Old target profiles/plans must be regenerated because operator graphs changed.

Pinned source snapshots and hashes remain in provenance.json.
These image workloads come from Cedar/DINO/SwAV/Transformers/BLIP, not five
Data-Juicer Hub recipes.

The frozen Hub adapters (pile_hackernews, pile_pubmed_abstracts, pile_freelaw,
pile_europarl, pile_uspto_backgrounds) remain available through the dispatcher
and hub_catalog.py; see hub_provenance.json.

## Validation

tests/test_explicit_features.py checks native SimCLRv2 equivalence, default
order equivalence to reference recipe callables, and actual Cedar execution
of sampled sparse orders. Tests establish execution and output contracts,
not downstream model accuracy equivalence.
