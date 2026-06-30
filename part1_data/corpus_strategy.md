# Part 1: Data Corpus — Strategy Write-up

The quality of the training corpus directly determines the quality of domain adaptation. Therefore, we implement a strategy which collects and filters domain specific data, which we can use later during CPT in part2.


## Why a Curated In-Domain Corpus

General LLMs underrepresent automotive E/E text in two ways: a domain knowledge gap and an out-of-distribution formatting gap (see README). Closing them requires in-domain text that is dense in. automotive E/E terminology. This part assembles and filters such a corpus for use in CPT (Part 2).


## Corpus Construction

Usable automotive E/E text spans several source types, which differ in depth, format and accessibility:

- **Reference articles:** Wikipedia gives broad conceptual coverage, clean prose and is openly licensed.
- **Academic literature:** Academic papers offer high technical depth but access is inconsistent (much is paywalled).
- **Community & code:** automotive GitHub repositories, ARXML examples and technical forums are noisy and license-mixed.
- **Vendor documentation:** tool and semiconductor vendors (e.g. CSS Electronics, NXP) publish freely accessible tutorials, reference manuals and datasheets, which are precise and domain-specific, but often copyrighted.
- **Standards & specifications:** the AUTOSAR specs and ISO 26262, for example, are authoritative, but  paywalled, copyrighted and formatting-heavy.
- **OEM & supplier documentation:** requirements, ARXML models and test specifications are rich in-domain text, but proprietary.

This toy setup uses the freely accessible, machine-parsable sources for demonstration: Wikipedia, vendor tutorials (e.g. CSS Electronics) and open-access papers. The most valuable E/E text, standards and proprietary OEM data, is excluded by the public-data constraint (see Limitations).

The pipeline also demonstrates a possible citation-driven expansion together with its limitations. Starting from handpicked seed papers (loaded via DOI), it downloads those references that are openly accessible via *OpenAlex*. This shows the corpus can be grown automatically from a few curated seeds instead of finding every source by hand.


## Preprocessing Pipeline

### Data Extraction

Different source types require different extraction approaches. **Plain text sources** can be used directly. **HTML pages** require stripping of navigation, scripts, and footers to isolate the main content. **PDFs** are the most common format for academic papers and technical standards but also the most problematic: text can be extracted via parser libraries, but embedded images, tables, diagrams and mathematical formulas are either lost or extracted as garbled characters. For formula-heavy documents, specialized extraction tools or manual preprocessing may be needed.

The extraction method determines what ends up in the corpus. Schema-heavy E/E documentation is particularly affected since much of its technical content is conveyed through diagrams and tables rather than prose.


### Language Filtering

Language filtering ensures corpus consistency by keeping only documents in the target language. The choice depends on the domain and intended training distribution. Automotive E/E documentation, for example, exists in both English and German, so filtering decisions should reflect which language coverage the model needs.


### Quality Filtering 

Quality filtering can be used to remove low-quality documents before chunking. Standard filters include word count thresholds to remove fragments and excessively long documents, mean word length to catch garbled text, symbol-to-word ratio to filter formula-heavy extractions, bullet-point ratio to remove low-information list pages and a stop-word check, since natural prose contains many common function words ("the", "and", "of") while keyword lists or garbled text do not (Rae et al. 2021).

Beyond general text quality, a relevance filter removes off-domain documents, keeping only text clearly related to automotive E/E.


### Deduplication

Exact deduplication via hash comparison removes identical documents. Deduplication matters because repeated text causes the model to over-memorize duplicated passages rather than generalize and wastes training compute on redundant data (Lee et al. 2022).


### Chunking

Documents are tokenized and split into fixed-length sequences of 2048 tokens. The model supports up to 128k tokens, but training at full length requires far too much memory. Therefore, a shorter fixed length is chosen as a memory/efficiency trade-off. Longer chunks preserve more document context but cost more per sequence: attention compute scales quadratically with length (Vaswani et al. 2017). The memory per sequence grows with length as well, so fewer sequences fit in VRAM at once.

### Train/Val Split

The final corpus can be split into train and validation sets. Validation loss during CPT serves as an early convergence signal without waiting for downstream task evaluation.


## Domain-Specific Vocabulary

Subword tokenizers may split rare words into multiple fragments. Since general tokenizers are trained on web text, automotive E/E terms like `AUTOSAR` or `0xF190` can fragment into pieces that carry little signal. This lengthens sequences and adds compute.

Two approaches can handle this:

#### Progressive embedding adjustment (used here)
The tokenizer is left unchanged and CPT gradually shifts the embeddings of the existing fragment tokens toward their automotive E/E meaning in context (Han & Han 2024).

#### Vocabulary extension (possible extension)
Dedicated tokens for frequent domain terms are added to the tokenizer, each initialized as the mean of the fragments it replaces (Hewitt 2021), then trained alongside CPT.


## Toy Setup

| Step | Implementation | Detail |
|---|---|---|
| Data Sources | Wikipedia API, web pages, open access seed PDFs | Wikipedia articles, web pages, 5 seed PDFs (via DOI) + reference PDFs crawled via OpenAlex (demo) |
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
| Raw collected (≈50 Wikipedia + 17 web sources + 5 seed papers + 7 citation-crawl PDFs) | 79 |
| After filtering (language, quality, relevance) | 71 |
| After deduplication | 71 |


Eight documents were filtered:
- **Relevance** removed 6: *Sensor fusion* (a general data-fusion article with no E/E terminology) plus five off-domain documents (mostly ML/robotics references pulled in by the citation crawl).
- **Low-prose** removed the OBD-II PID table page.
- **Language** removed the German CAN page.

Deduplication found no exact duplicates this run. The word-count, non-ASCII and length filters did not trigger.

The 71 unique documents were chunked into 245 sequences of 2048 tokens (~430,000 tokens total), split 220 train / 25 validation.

A per-document breakdown (word count, stop-word ratio, non-ASCII ratio, E/E-keyword count, filter status) is written to `corpus_stats.csv` and rejected documents are moved to a `filtered/` folder.

**Citation crawl:** Of 130 unique references resolved across the five seeds, 19 had a direct open-access PDF link and only 7 returned an actual PDF file. This demonstrates the mechanism while showing it does not scale in the toy setup.

**Seed papers:** Bock (automotive software taxonomy), Cuomo (RISC-V ECU platform), Mauser (E/E architecture centralization), Salay (functional safety), Ulbrich (system architecture for automated driving).


## Limitations

This toy prototype demonstrates the full corpus construction pipeline from source fetching through tokenized chunking, but several gaps separate the outcome from a production-scale training corpus. 

| Limitation | Impact | Production fix |
|---|---|---|
| Small data corpus (~430k tokens, ~245 chunks) | The model will not get enough exposure to the automotive E/E concepts to deliver production scale results. | Expand the corpus with more curated sources (additional Wikipedia/web pages, internal datasets) and where licensing allows, automated reference following. |
| Source licensing | Publicly accessible sources vary in license and some cannot be freely reused at production scale. | Obtain license clearance or use in-house data. |
| Citation-following hard to scale | Most cited papers are paywalled, open-access PDF links are inconsistent, and APIs are rate-limited. | Get proper access to the papers through publisher APIs or subscriptions. |
| Image and table loss (pypdf) | A lot of important content that distinguishes automotive E/E text from general prose is never seen by the model, because pypdf silently drops all non-text content. | Use a richer extraction pipeline: convert tables to structured Markdown and add a vision model in preprocessing to caption images into text. |
| Tokenizer vocabulary mismatch | Domain-specific notation may be split suboptimally and into many fragments. This lengthens sequences and spreads a term's meaning across many token positions. | Evaluate whether tokenizer extension improves downstream perplexity and task performance. |
| No near-duplicate detection | The same papers could appear slightly different multiple times in the corpus and inflate the effective training signal. | Implement near-duplicate detection for example via MinHash. |
| Fixed 2048-token chunks, no overlap | Documents are cut at hard token boundaries, so sentences or concepts spanning a boundary are split across two chunks and lose their connecting context. | Add chunk overlap and/or structure-aware chunking so semantic units stay intact. |



## References

- Rae, J. W. et al. (2021). *Scaling Language Models: Methods, Analysis & Insights from Training Gopher.* arXiv:2112.11446
- Lee, K. et al. (2022). *Deduplicating Training Data Makes Language Models Better.* ACL 2022. arXiv:2107.06499
- Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS 2017. arXiv:1706.03762
- Han, D. & Han, M. (2024). *Continued Pretraining with Unsloth.* unsloth.ai/blog/contpretraining.
- Hewitt, J. (2021). *Initializing New Word Embeddings for Pretrained Language Models.* cs.columbia.edu/~johnhew/vocab-expansion.html.