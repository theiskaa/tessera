# Phase 015: Audit real evaluation source coverage

Status: complete and reviewed. This phase does not change the model, training input, gold labels, or site.

The frozen 877-document detector development set was matched by exact `name` to raw records. All 877 matched one record, with identical text and country and a URL. The V17 silver set's 12,373 documents also matched raw IDs; 367 labels use excerpts rather than the full raw text, which is expected. No gold document text exactly matches any silver training document text. URL hosts were normalized only by dropping `www.`; host overlap is a conservative proxy, not verified publisher identity.

| Country | Gold docs | Gold URL hosts | V17 silver hosts | Gold docs on a silver host | Address labels on a silver host |
| --- | ---: | ---: | ---: | ---: | ---: |
| US | 101 | 2 | 1 | 100 | 115/116 |
| GB | 152 | 65 | 25 | 50 | 57/144 |
| DE | 117 | 41 | 253 | 0 | 0/255 |
| GE | 186 | 44 | 43 | 0 | 0/311 |
| JP | 163 | 49 | 16 | 0 | 0/328 |

The US development set is almost entirely Federal Register pages; V17 silver contains Federal Register pages too. This tests different documents from the same host, not new-host transfer. The GB same-host rows are `gov.uk`. Other government subdomains may still share a publisher despite different hosts, so the zeroes do not certify fully independent publisher holdouts. GE and JP address-focused collections span 37 and 43 hosts respectively, but their top five hosts carry 164/282 and 181/302 labels. Country totals remain sensitive to those hosts.

The private script `internal/bench/review/audit_gold_hosts.py` writes an aggregate audit and a host-annotated gold copy under ignored paths. A separate check confirmed all 877 annotated rows retain the same order, text, and labels; only `source_host` was added, as `COUNTRY|host` so a shared host is not pooled across countries. The annotated gold SHA-256 is `e283b6db04469fc67945f0204a597fe25da092fa2f91062073325f1213028887`; the original frozen gold remains `3940db21008a51dc71e78fdb390b1be3b1f69ca4596b4c49b50da0af77b2abbd`.

`trainer/src/detect_eval.rs` accepts the optional field as a JSON report slice without passing it to the model. The Markdown report still prints the existing country and document-type tables. A release trainer build and formatting check passed. Fresh V17 evaluations on original and annotated gold had identical overall, per-kind, per-country, per-document-type, per-script, and false-positive score objects. The same equality holds for V18 versus its earlier unannotated run. A paired URL-host bootstrap self-check compared V17 with itself and produced exactly zero for all 15 intervals. Its V17/V18 diagnostic confirms wide intervals for address changes, for example JP exact −17.3 to +1.6 points and GE exact −11.2 to +2.0. The frozen point-estimate guard still rejects V18; the intervals prevent unsupported claims about every publisher. Raw outputs: ignored `internal/reports/m7/gold-host-audit.json`, `host-bootstrap-selfcheck.md`, and `v18-host-bootstrap.md`.

Release implication: the still-needed sealed new-publisher set in phase 002 is especially urgent for US and GB. No per-country target should be called externally verified from this development set.
