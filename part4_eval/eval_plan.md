# Part 4: Evaluation — Eval Plan

Evaluation measures whether the CPT and SFT actually improve the model on the domain-specific tasks. We compare the three stages, **Base → CPT → CPT+SFT**, on the same automotive E/E questions. For the toy setup the eval is specified as a plan and no full run is executed.


## Evaluation Set

10 questions with reference answers in two groups.


### Near (covered)

These are reformulations of SFT Q&A pairs. They should test whether the model generalizes the instruction format beyond memorized strings.

1. Q: *In CAN, which message wins the bus when several nodes transmit at the same time and why?*  
  A: The message with the lowest identifier has the highest priority, because a dominant bit (0) overrides a recessive bit (1).
2. Q: *List the four ASIL levels defined by ISO 26262, from lowest to highest.*  
  A: 
  - ASIL A
  - ASIL B
  - ASIL C
  - ASIL D
3. Q: *How does CAN FD achieve higher throughput than classical CAN?*  
  A: Up to 64 data bytes per frame instead of 8, plus a higher bit rate in the data phase via bit-rate switching, while arbitration stays at the classical rate.
4. Q: *What is the role of the AUTOSAR Runtime Environment (RTE)?*  
  A: RTE is the middleware between application software components and the Basic Software. It implements standardized communication and hides the ECU hardware and bus.
5. Q: *Which OBD-II service reports stored Diagnostic Trouble Codes?*  
  A: Service 03 shows stored Diagnostic Trouble Codes.

### Far (uncovered)

These include topics which were only present in the CPT corpus and absent from the SFT pairs, to test whether the model falls back on CPT knowledge or hallucinates.

6. Q: *What is brake-by-wire?*  
  A: A braking system that replaces the mechanical/hydraulic link between pedal and brakes with electronic signals and actuators.
7. Q: *What does an anti-lock braking system (ABS) do?*  
  A: It prevents the wheels from locking up during braking, so the driver keeps steering control.
8. Q: *What is the purpose of an airbag?*  
  A: It inflates within milliseconds during a collision to cushion the occupants and then deflates again.
9. Q: *What does a Tire-Pressure Monitoring System (TPMS) do?*  
  A: It monitors the air pressure inside the tires and warns the driver when a tire's pressure is too low.
10. Q: *What is eCall?*  
  A: An EU system that automatically calls emergency services and sends the vehicle's location after a collision.


## How to Compare Base vs. CPT vs. CPT+SFT (fairly)

All three stages receive the **identical question in the same Alpaca prompt** with **greedy decoding (temperature 0)** for deterministic, reproducible output. The base model is given the same prompt even though it is not instruction-tuned. This makes the contribution of SFT visible, because the base model has no answer format.

Different metrics can be used to compare all stages:

- **Manual rating:** captures factual accuracy and answer format, which is what matters for open-ended Q&A.
- **LLM-as-judge:** a larger model scores each answer using the reference as an anchor (reference-guided). A strong LLM judge reaches ~80% agreement with human raters (Zheng et al. 2023).
- **ROUGE (1/L, F1):** automatic n-gram overlap with the reference (Lin 2004). This gives one comparable number per stage, but it rewards lexical overlap, not correctness.


## Toy Setup

| Stage   | Model                               |
|---------|-------------------------------------|
| Base    | `unsloth/Llama-3.2-1B` (16-bit)     |
| CPT     | base + CPT adapter, merged (16-bit) |
| CPT+SFT | CPT+SFT, merged (16-bit)            |

Models are loaded sequentially (T4 memory), asked the same 10 questions, greedy decoding.


## Expected Outcomes (not executed)

#### Base → CPT
Domain vocabulary and coherent style appear, but no instruction format. So the model continues the text instead of answering in the expected format.

#### CPT → CPT+SFT 
Structured, instruction-following answers appear. 
- **Near** questions should benefit most, while factual accuracy changes little.
- **Far** questions test whether the model uses CPT knowledge or hallucinates. At the toy scale, factual errors are expected (consistent with the Part 2 and Part 3 findings).

So SFT should add the format, while CPT/corpus scale bounds accuracy.


## Limitations

| Limitation | Impact | Production fix |
|---|---|---|
| Small eval set (10 questions) | High variance | Scale to a larger question set. |
| ROUGE validity | Rewards lexical overlap, not factual correctness. | Add factuality / LLM-judge metrics. |
| One reference answer per question | Multiple valid phrasings are penalized. | Allow multiple references or semantic scoring. |
| Manual rating | Subjective and not reproducible at scale. | Use multiple annotators or an LLM judge. |
| LLM-judge bias | An LLM judge has position, verbosity and self-enhancement biases. | Swap orderings, reference-guided grading, average judges. |


## References

- Lin, C.-Y. (2004). *ROUGE: A Package for Automatic Evaluation of Summaries.* ACL 2004. aclanthology.org/W04-1013
- Zheng, L. et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* arXiv:2306.05685