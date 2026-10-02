# Data analysis and recommended experiments

The current data are sufficient for basic implementation and reproducible preliminary runs. They are not yet a strong deployment benchmark. All raw checksums match the imported manifest, and the processed corpus contains no normalized exact duplicates, including the separate hard-negative file.

## What the audit reveals

| Source | Retained rows | Median characters | Role |
| --- | ---: | ---: | --- |
| Alpaca | 1,260 | 68 | Assumed benign instructions |
| DAN | 1,356 | 1,762.5 | Human-written jailbreak templates, including 196 held-out rows |
| JBB harmful | 100 | 81.5 | Direct harmful requests |
| JBB benign | 100 | approximately 73 | Separate matched benign test |

The length rule achieves **0.906 test F1**, **0/15 JBB harmful test recall**, and **196/196 held-out recall**. Thus the community holdout is still vulnerable to a length shortcut. An all-positive test cannot measure false positives.

The training-fitted character TF-IDF audit finds a nearest training cosine similarity of at least 0.90 for **47/174 DAN test prompts** (37 at 0.95). There are **45 labeled communities shared between training and test**. The held-out split has no matches above 0.90. This diagnostic indicates template similarity; it does not assign semantic duplicate labels.

Alpaca's overlap filter removed zero rows. That is not a safety audit. There are also two positive concepts in the corpus: direct harmfulness and an attempt to bypass instructions. A future label schema should distinguish them even if a deployment decision ultimately combines them.

## Architecture comparison

Use the **same dataset condition for every architecture**, with the same row identifiers and splits. Keep separate score slices for each source. Comparing an MLP trained on Alpaca/DAN with a transformer trained on WildGuard would say little about architecture.

The first condition is the imported corpus, preserved so the initial results remain reproducible. Its limitations should accompany every result table. The expanded condition below adds longer benign prompts and keeps variants of each underlying request together. Further work should group semantic near-duplicates, review ambiguous labels, and add realistic user prompts. Apply each new condition to all four architectures. Match context budgets and report remaining representation differences.

Separate training on an injection dataset is worthwhile **as a separate task condition for each model**, rather than one dataset per architecture. Transfer from the original jailbreak corpus to injection text can also be reported, but failure there may reflect a label mismatch. Neither experiment alone demonstrates indirect-injection detection in a real application.

## Expanded dataset condition

WildJailbreak is now acquired at revision `5ddc12a7894f842b0619b8e1c7ee496b198af009` after provider access was enabled. The downloaded training file contains 261,538 rows. Its four categories provide both short vanilla and longer adversarial examples in each class, reducing the original reliance on short Alpaca negatives and long DAN positives. Completions are not model inputs, and a refusal does not change the prompt's harmfulness label.

The expanded comparison uses exactly 100,000 unique training prompts, 39,502 validation prompts, 38,911 test prompts, and 2,210 official evaluation prompts. Variants of the same normalized vanilla request remain in one main partition. Exact normalized duplicates are removed globally, conflicting prompt labels are excluded, and evaluation prompts take precedence over training duplicates. The official evaluation omits the vanilla request, so underlying-request overlap with that set cannot be assessed. Main-split overlap and file checksums are recorded in `results/large/data_audit.json` and `data_manifest.json`.

Training is capped from an eligible pool of 183,005 rows to give every architecture the same 100,000-row budget. No repetitions or duplicated versions are added to reach the target. The length-only test F1 is 0.694, compared with 0.906 on the earlier corpus. Length still predicts some labels, and grouping exact underlying requests does not eliminate shared tactics or paraphrases. Upstream labels and the largely synthetic origin remain limitations.

WildJailbreak defines a prompt harmfulness task. Prompt injection can occur in text that is otherwise harmless, so injection datasets should form a separate condition with context and their own label definition. Response-safety and preference datasets are not suitable substitutes for prompt labels. Dataset version totals must not be added together without checking overlapping records.

## Other candidate sources

| Priority | Source | Recommended use | Checks before merging |
| --- | --- | --- | --- |
| 1 | [Shen regular prompts](https://huggingface.co/datasets/TrustAIRLab/in-the-wild-jailbreak-prompts/tree/main/regular_2023_12_25) | First an external false-positive check; then reviewed benign candidates in a second condition | Non-jailbreak does not mean harmless. Check overlap, templates, language and length. |
| 2 | [deepset prompt injections](https://huggingface.co/datasets/deepset/prompt-injections) | Separate injection condition using its existing train/test roles | Injection labels describe redirection, not general harmfulness. Inspect duplicates and context assumptions. |
| 3 | [WildGuardMix](https://huggingface.co/datasets/allenai/wildguardmix) | Larger prompt-harm condition using train for training and test for evaluation | Requires provider-granted access. Exclude missing prompt-harm labels; group repeated prompts paired with different responses; check overlaps. |

The authors' [DAN repository](https://github.com/verazuo/jailbreak_llms) documents both regular and jailbreak releases. The deepset card advertises Apache-2.0 and currently lists 662 examples across its partitions, so it is useful as a small task-specific experiment rather than a way to reach a 10,000-prompt target. WildGuard's card distinguishes prompt harmfulness from its adversarial flag and response labels; those fields must not be substituted for one another.

Do not merge the 107,250 evaluation-question file as attack templates. Do not double-count AdvBench behaviors already present in JBB. Do not consume HarmBench test behaviors as extra training positives while also calling the resulting evaluation external.

Candidate-source documentation was checked on 2 October 2026. The earlier optional-source acquisition attempt failed on DNS resolution. Network access now works, and WildJailbreak supplies the larger comparison; the optional extension candidates have not been merged into it. An AIDataFdn prompt-injection CSV was downloaded during source screening but is not used for training: its different label definition and limited provenance would make it a poor silent addition to this condition.

## Next steps

Compare the expanded four-model results on the shared partitions, including each data-type slice and the official benign acceptance rate. Preserve the earlier comparison separately. Add reviewed natural user prompts and a context-aware injection condition for subsequent transfer tests. Repeat promising configurations across seeds before drawing stronger architecture conclusions.
