# Model and limits

[README](../README.md) · [Getting started](getting-started.md) · [Development](development.md)

## The pipeline

Tessera combines two trained networks with deterministic contact rules.

1. Find email and phone spans with rules. Those spans are protected from competing model labels.
2. Tokenize the source and compute character n-gram, script, shape, and contextual features.
3. Run the detector to identify people, organizations, and addresses.
4. Run the address parser on detected addresses to label components such as road, city, and postcode.
5. Group related entities into contacts. Details with ambiguous ownership remain `unassigned`.

The bundle contains both the parser and detector. Email and phone extraction can run with no bundle. The rules layer also normalizes email domains and phone numbers; names and postal addresses retain their source text.

## US detector architecture

The US context detector uses hashed character n-grams of lengths 2, 3, and 4, together with script, shape, and flag features. A convolutional network predicts seven BIO labels: `O`, and begin/inside labels for PERSON, ORG, and ADDRESS.

| Setting | Value |
| --- | --- |
| Architecture | `detector-context96-rms-v2` |
| Hidden channels | 96 |
| Convolution kernel | 3 |
| Dilations | `[1, 2, 4, 8, 16, 1, 64]` |
| Normalization | Channel RMS after each residual block, epsilon `1e-5` |
| Context radius | 96 retained tokens on either side |
| Inference window | 2,048 retained tokens |
| Window overlap | 448 retained tokens |

RMS normalization divides each token's channel vector by its root mean square, with epsilon for numerical stability. It adds no learned parameters. The address decoder can continue an address across an internal low-confidence dip.

The overlap and context margins ensure a span of at most 256 retained tokens fits inside a trusted window. This is a coverage guarantee; a covered span can still be classified incorrectly.

Format 2 bundles bind this graph and decoder explicitly. The reader also supports legacy format 1 bundles with their original six-block graph. The filename `tessera-v1.safetensors` identifies the distributed asset; its embedded metadata records the model version and format.

## Measured quality

The latest US native checkpoint completed 4,000 optimizer updates. The following results use exact entity F1: both the entity kind and its boundaries must match. Scores are percentages, measured on the native checkpoint before packaging.

| Evaluation set | Documents | PERSON | ORG | ADDRESS |
| --- | ---: | ---: | ---: | ---: |
| Real training-seen documents | 1,799 | 96.47 | 75.39 | 97.33 |
| Reused real development set | 196 | 76.86 | 53.76 | 89.22 |

Training-seen results measure how well the model learned examples it was trained on. The development set was reused during model selection. Neither row establishes accuracy on fresh unseen documents.

The model failed the automatic promotion requirements, including the full 95% training-seen gate and some group and negative-control checks. It is offered as experimental with those limits retained. Aggregate PERSON results also conceal weaker biography coverage. Organization names, aliases, and boundaries remain a substantial source of error.

These scores evaluate the detector's PERSON, ORG, and ADDRESS spans. They do not measure email rules, phone rules, address-component accuracy, or whether contact assignments are correct.

The small public detector fixture suite currently has eight documented failing cases out of thirteen. The expected labels are retained, including missed people in prose and tables. These examples expose known limitations; they are not an independent accuracy benchmark or a claim that the release meets 95% everywhere.

The packaged address parser was separately checked through the public `parse_address` API on its 3,000-row US test shard: component F1 was **99.05%**, and **95.23%** of addresses had every component exactly right. This tests parsing a supplied address, not finding addresses in arbitrary documents.

## Distributed artifact

| Property | Value |
| --- | --- |
| Model version | `0.3.0` |
| Runtime package version | `0.1.0` |
| Bundle format | `2` |
| Bundle size | 3,489,616 bytes (3.49 MB before compression) |
| Weight encoding | Per-channel symmetric int8; float32 scales and biases |

The 4,000-update bounded run finished. It did not complete the configured full training schedule, so the metadata retains `detector_training_complete=false`. The original checkpoint and failed quality gates are preserved.

The Safetensors header contains 39 metadata fields, including network dimensions, dilations, labels, feature settings, source hashes, experimental status, and licensing. [bundle.json](../models/bundle.json) exposes the same metadata in a readable file. [tessera-v1.sha256](../models/tessera-v1.sha256) is the checksum to pass when loading the bundle.

## Using results

Treat model confidence and `review_recommended` as signals for inspection, rather than an accuracy guarantee. Check names and organizations before using them to populate important records. A wrong or missing anchor can affect contact grouping as well as entity detection.

Phone hints support interpreting national numbers; they do not establish named-entity coverage for that country. The current detector work focuses on US material. Other languages and document styles have no demonstrated general accuracy from this evaluation.

## Data and attribution

The training pipeline combines synthetic contact documents with annotated real US source documents. The address parser also has its own training history. Hashes, source membership, and evaluation provenance are kept in local training artifacts; the public repository excludes raw corpora and run directories.

Code is MIT OR Apache-2.0. [NOTICE](../NOTICE) declares CC BY 4.0 for trained weights and preserves the source attributions, including OpenStreetMap's ODbL address data and Open Addresses UK's notices. Redistribution should retain those notices and the terms applicable to the underlying data.
