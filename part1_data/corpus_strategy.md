# Part 1: Data Corpus — Strategy Write-up

The quality of the training corpus directly determines the quality of domain adaptation. Therefore, we implement a strategy which collects and filters domain specific data, which we can use later during CPT in part2.


## Why a Curated In-Domain Corpus

General LLMs underrepresent automotive E/E text in two ways — a domain knowledge gap and an out-of-distribution formatting gap (see README — Motivation). Closing them requires in-domain text that is dense in E/E terminology and specification formats. This part assembles and filters such a corpus for use in CPT (Part 2).

Continued Pre-Training (CPT) on in-domain text addresses both gaps at once: in our case it does not add new tokens to the "vocabulary" of the tokenizer, but shifts the embeddings of existing tokens toward their automotive E/E meaning and learns the domain context that links them through attention, shifting the model's token distribution toward automotive E/E patterns.


## Corpus Construction

Usable automotive E/E text comes from different categories of public sources. The three main categories used in this toy setup are **academic papers**, **reference articles (Wikipedia)** and **domain-specific web pages**. Each contributes differently: papers provide precise technical depth, Wikipedia broad conceptual coverage and web pages accessible explanations.

The biggest part of the corpus in this toy example is built from a curated set of Wikipedia articles and technical web pages, selected from core E/E concepts. This keeps relevance density high and makes collection fast and reproducible.

On top of this, the pipeline demonstrates a possible citation-driven expansion together with its limitations. Starting from handpicked seed papers (loades via DOI), it downloads those references that are openly accessible via OpenAlex. This shows the corpus can be grown automatically from a few curated seeds instead of finding every source by hand.


## Preprocessing Pipeline

### Data Extraction

Different source types require different extraction approaches. **Plain text sources** can be used directly. **HTML pages** require stripping of navigation, scripts, and footers to isolate the main content. **PDFs** are the most common format for academic papers and technical standards but also the most problematic: text can be extracted via parser libraries, but embedded images, tables, diagrams and mathematical formulas are either lost or extracted as garbled characters. For formula-heavy documents, specialized extraction tools or manual preprocessing may be needed.

The extraction method determines what ends up in the corpus. Schema-heavy E/E documentation is particularly affected since much of its technical content is conveyed through diagrams and tables rather than prose.


### Language Filtering

Language filtering ensures corpus consistency by keeping only documents in the target language. The choice depends on the domain and intended training distribution. Automotive E/E documentation, for example, exists in both English and German, so filtering decisions should reflect which language coverage the model needs.


### Quality Filtering 

Quality filtering can be used to remove low-quality documents before chunking. Standard filters include word count thresholds to remove fragments and excessively long documents, mean word length to catch garbled text, symbol-to-word ratio to filter formula-heavy extractions, bullet-point ratio to remove low-information list pages and a stop-word check, since natural prose contains many common function words ("the", "and", "of") while keyword lists or garbled text do not (Rae et al. 2021).

Beyond general text quality, a relevance filter removes off-domain documents that slip in through citation lists, keeping only text clearly related to automotive E/E.


### Deduplication

Exact deduplication via hash comparison removes identical documents. Deduplication matters because repeated text causes the model to over-memorize duplicated passages rather than generalize and wastes training compute on redundant data (Lee et al. 2022).


### Chunking

Documents are split into fixed-length token sequences to match the model's training context window and to achieve lower memory usage. Longer chunks preserve more document context but cost more per sequence — attention compute scales quadratically with length (Vaswani et al. 2017) and memory grows with length — so fewer sequences fit in VRAM and the effective batch size drops.


### Train/Val Split

The final corpus can be split into train and validation sets. Validation loss during CPT serves as an early convergence signal without waiting for downstream task evaluation.


## Domain-Specific Vocabulary

Subword tokenizers may split rare words into multiple fragments. Since general tokenizers are trained on web text, Automotive E/E terms like `AUTOSAR` or `0xF190` can fragment into pieces that carry little signal. This lengthens sequences and adds compute.

Two approaches can handle this:

#### Progressive embedding adjustment (used here)
The tokenizer is left unchanged and CPT gradually shifts the embeddings of the existing fragment tokens toward their automotive E/E meaning in context. No structural change and sufficient at this corpus scale (Han & Han 2024).

#### Vocabulary extension (possible extension)
Dedicated tokens for frequent domain terms are added to the tokenizer, each initialized as the mean of the fragments it replaces (Hewitt 2021), then trained alongside CPT.


## Toy Setup

| Step | Implementation | Detail |
|---|---|---|
| Data Sources | Wikipedia API, web pages, open access seed PDFs | Wikipedia articles, web pages, 5 seed PDFs + reference PDFs crawled via OpenAlex (demo) |
| Text Extraction | pypdf, wikipediaapi, BeautifulSoup4 | HTML: strips script, style, nav, footer, header tags |
| Language Filtering | langdetect | English only |
| Quality Filtering | Word count + non-ASCII ratio + stop-word ratio | < 50 or > 100,000 words removed; non-ASCII > 10% removed; stop-word ratio < 10% removed (filters table/keyword-list pages) |
| Relevance Filtering | Keyword check | ≥ 3 E/E keywords required |
| Deduplication | SHA256 hashing | Exact duplicates get removed |
| Chunking | 2048 tokens | Llama-3.2-1B tokenizer, no overlap |
| Train/Val Split | 90/10 | Fixed index split |


## Results

Running the pipeline end-to-end produced the following corpus:

| Stage | Documents |
|---|---|
| Raw collected (≈49 Wikipedia + 15 web pages + 5 seed PDFs + 7 citation-crawl PDFs) | 76 |
| After filtering (language, quality, relevance) | 69 |
| After deduplication | 69 |

Seven documents were filtered, demonstrating the filters:

- **Relevance** removed 5 documents: *Sensor fusion* (a general data-fusion article with no E/E terminology) and four off-domain reference PDFs.
- **Low-prose** removed the OBD-II PID table page (stop-word ratio 0.06, below the 0.10 threshold).
- **Language** removed a German Wikipedia page on CAN, deliberately included to demonstrate the filter.

Deduplication found no exact duplicates this run. The word-count, non-ASCII and length filters did not trigger.

The 69 unique documents were chunked into 213 sequences of 2048 tokens (367,374 tokens total), split 191 train / 22 validation.

A per-document breakdown (word count, stop-word ratio, non-ASCII ratio, E/E-keyword count, filter status) is written to `corpus_stats.csv`, and rejected documents are moved to a `filtered/` folder for inspection.

**Citation crawl yield.** Of 130 unique references resolved across the five seeds, 19 had a direct open-access PDF link and only 7 returned an actual PDF file. This demonstrates the mechanism while showing it does not scale in the toy setup.

**Seed papers:** Bock (2016, automotive SW taxonomy); Cuomo (2023, RISC-V ECU platform); Mauser (2024, E/E centralization); Salay (2017, ISO 26262 / functional safety); Ulbrich (2017, functional system architecture for automated vehicles).


## Limitations

This toy prototype demonstrates the full corpus construction pipeline from source fetching through tokenized chunking, but several gaps separate it from a production-scale training corpus. 

| Limitation | Impact | Production fix |
|---|---|---|
| Small data corpus (~367k tokens, ~213 chunks) | A production CPT corpus for a 1B-param model requires a much bigger amount of tokens. At this scale, perplexity reduction is measurable but the model does not get enough exposure to the automotive E/E concepts to deliver production scale results. | Expand the corpus with more curated sources (additional Wikipedia/web pages, internal datasets) and, where licensing allows, automated reference following. |
| Citation-following hard to scale | The reference crawl works as a demonstration but most cited papers are paywalled, open-access PDF links are inconsistent, and APIs are rate-limited. | Get proper access to the papers through publisher APIs or subscriptions and cache what is already downloaded so the APIs are not hit repeatedly. |
| Image and table loss (pypdf) | Automotive E/E documentation is schema-heavy and important parts are embedded as images or vector graphics. pypdf silently drops all non-text content. Therefore, a lot of structural and quantitative content that distinguishes automotive E/E text from general prose is never seen by the model. | Replace pypdf with a better pipeline that converts tables to structured Markdown, which a text-only model can learn directly. Figures and diagrams cannot be ingested by a text model at all. They require a vision model in preprocessing to generate textual descriptions (captioning/OCR) that the text model then learns. |
| Formula and notation loss | The non-ASCII filter is necessary because pypdf extracts mathematical formulas as garbage character sequences which would inject noise directly into the model weights. However, the same filter also silently discards meaningful E/E notation like UDS byte sequences (0x7DF 0x03 0x22 0xF1 0x90). The trade-off is accepted at toy scale. | Implement a notation whitelist: before applying the non-ASCII filter, identify and preserve patterns matching for example byte-sequence notation. Use LaTeX-aware PDF extraction for formula-containing documents. |
| Tokenizer vocabulary mismatch | The Llama-3.2-1B tokenizer was trained on general web text. Subword splitting is normal but domain-specific notation can be split suboptimally and into many fragments. This lengthens sequences and spreads a term's meaning across many token positions. | At full scale, evaluate whether tokenizer extension improves downstream perplexity and task performance. This is as a concrete production step for highly specialized domains (McCormick 2025). |
| No near-duplicate detection | Exact deduplication via SHA256 hashing catches identical documents but not near-duplicates. At production scale with a lot more documents, the same papers could appear slightly different multiple times in the corpus. This would inflate the effective training signal from multiple times appearing documents. | Implementing near-duplicate detection for example via MinHash. |
| Fixed 2048-token chunks, no overlap | Documents are cut at hard token boundaries, so sentences or concepts spanning a boundary are split across two chunks and lose their connecting context. | Add chunk overlap and/or structure-aware chunking so semantic units stay intact. |


## References

- Rae, J. W. et al. (2021). *Scaling Language Models: Methods, Analysis & Insights from Training Gopher.* arXiv:2112.11446
- Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS 2017. arXiv:1706.03762
- Lee, K. et al. (2022). *Deduplicating Training Data Makes Language Models Better.* ACL 2022. arXiv:2107.06499
- McCormick, C. (2025). *Continuing Pre-Training on Raw Text.* mccormickml.com.
- Han, D. & Han, M. (2024). *Continued Pretraining with Unsloth.* unsloth.ai/blog/contpretraining.
- Hewitt, J. (2021). *Initializing New Word Embeddings for Pretrained Language Models.* cs.columbia.edu/~johnhew (web note).