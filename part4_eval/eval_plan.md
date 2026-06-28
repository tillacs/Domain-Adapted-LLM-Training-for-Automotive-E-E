# Part 4: Evaluation — Eval plan

Evaluation measures whether domain adaptation actually improves the model.
We compare three stages: ``Base → CPT → CPT+SFT``on the same E/E questions.

## Evaluation set

10 questions with known reference answers in two groups:

- **Near(covered concepts):** reformulations of SFT training topics (RTE, CAN arbitration, ASIL, CAN-FD): tests whether the format generalizes beyond memorized strings.

- **Far (uncovered concepts):** topics not in the SFT set (FlexRay, TSN, UDS 0x2E, WdgM): tests whether the model falls back on CPT knowledge or hallucinates.


## How to compare Base Model vs. CPT Model vs. CPT+SFT Model

All three stages receive the identical question in the same Alpaca prompt, with greedy decoding (temperature 0) for deterministic, reproducible outputs. Two metrics:

- **ROUGE-L:** automatic n-gram overlap with the reference; one comparable number per stage. Limitation: rewards surface overlap, not factual correctness.

- **Manual rating (correct / partial / wrong):** captures factual accuracy and answer format, which is essential for open-ended Q&A.

**?- an other possibility would be to give a big model the training data as acontext and let it rate the aswers?**


## Toy Setup

| Stage   | Model                               |
|---------|-------------------------------------|
| Base    | unsloth/Llama-3.2-1B (16-bit)       |
| CPT     | base + CPT adapter, merged (16-bit) |
| CPT+SFT | merged SFT model   **?auch 16bit?** |

Models loaded sequentially (T4 memory), same 10 questions, greedy decoding.


## Results



[ROUGE-L per stage + mean table]
▎
▎ Pattern:
▎ - Base → CPT: domain vocabulary and coherent style appear, but no instruction format (the model continues text rather than answering).
▎ - CPT → CPT+SFT: structured, instruction-following answers appear; factual accuracy changes little.
▎
▎ Key finding: SFT adds the format, CPT adds the knowledge. Factual errors after CPT+SFT (e.g. ASIL-A/D inverted) trace to CPT/base knowledge and 1B scale, not the SFT step — even a trained concept is answered wrongly.


The CPT+SFT model is evaluated qualitatively on two prompt sets, neither seen in its exact training phrasing:

- **Near:** reformulations of covered concepts: rephrasing of training topics (RTE, CAN arbitration, ASIL, CAN-FD). Correct answers here indicate the model generalized the instruction-response format rather than memorizing exact training strings.

- **Far:** uncovered concepts: topics absent from the training set (FlexRay, TSN/automotive Ethernet, UDS 0x2E, WdgM). These probe whether the model falls back on CPT domain knowledge or hallucinates beyond its coverage.

All generations use greedy decoding (temperature 0) for deterministic comparison. [Observations to be filled in after the run.]






**Qualitative before/after.** On all three probe prompts the base model degenerates into repetition loops with no domain content. After CPT the loops disappear and the model produces coherent domain-style text using correct E/E vocabulary (ECUs, functional domains, service-oriented architecture, the functional-safety lifecycle). Factual precision, however, remains limited: the post-CPT CAN description correctly situates CAN among powertrain/chassis/body domains but wrongly attributes a "central controller" to what is a multi-master bus. This is the expected behavior of CPT — it shifts the token distribution toward domain style and vocabulary but does not reliably encode precise facts at this model and corpus scale.


## Limitations:

| Limitation | Impact | Production fix |
|---|---|---|
| Small eval set |  |  |
| ROUGE |  rewards lexical overlap not correctness |  | 
| One reference answer per question; multiple valid phrasings possible |  |  | 
|  |  |  | 


## References

- Lin, C.-Y. (2004). ROUGE: A Package for Automatic Evaluation of Summaries. ACL 2004.
