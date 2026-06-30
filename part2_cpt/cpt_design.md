# Part 2: Continued Pre-Training — Design Write-up

Continued Pre-Training (CPT) adapts a general base model to a target domain by continuing next-token training on in-domain text. The goal is to turn a generic model into one that has internalized automotive E/E patterns before any instruction tuning (McCormick 2025).

The base Llama-3.2-1B variant is used (see README). CPT on raw text could otherwise degrade an instruct model's existing instruction-following.


## Training Objective

CPT continues the model's original causal language-modeling objective *next-token prediction*: given a sequence of tokens x₁, x₂, …, xₙ₋₁, the model predicts xₙ.

The loss is cross-entropy averaged over all token positions in the chunk. Every token in the corpus simultaneously serves as both input context and training target and no manual labeling is needed. This is called *self-supervised learning*.

The model shifts its probability distribution toward automotive E/E text by repeatedly predicting the next token in domain-specific sequences, with the cross-entropy loss backpropagated to update the weights.



## Why CPT Before SFT

Domain-Adaptive Pre-Training (DAPT) gains scale with domain distance from the pretraining corpus. Domains well-represented in web text benefit little, while structurally absent domains benefit substantially (Gururangan et al. 2020).

Automotive E/E is a narrow engineering niche with its own terminology and specification formats that have no meaningful presence in most of the general web text.

For example, without CPT, the model may be more likely to interpret `CAN` as the English verb or noun than as *Controller Area Network*.

Further, a small set of instruction pairs teaches the instruction format on top of existing knowledge but should not build the new domain knowledge (Zhou et al. 2023). Because the base model lacks this knowledge for a niche domain like automotive E/E, it has to be acquired first through CPT.


## Learning Rate Strategy

The learning rate strategy for CPT addresses different stability concerns.

#### Learning Rate Schedule
CPT begins from a well-optimized pretrained checkpoint, not from random initialization. A full learning rate at step 0 risks destroying attention patterns developed during pretraining. A short linear warmup phase brings the learning rate to its target value gradually, keeping early updates small to preserve the pretrained structure. After warmup, a cosine schedule decays the learning rate smoothly from its peak toward zero.

#### Decoupled Learning Rate
Finally, not all model components require the same degree of adaptation. Vocabulary embeddings encode the most compressed form of semantic knowledge from pretraining. Applying the same update magnitude as attention and FFN layers risks destabilizing this structure. A decoupled, smaller embedding learning rate allows domain-specific token meaning to shift gradually without destroying the general semantic prior (Han & Han 2024).


## Avoiding Catastrophic Forgetting

CPT must update the vocabulary embeddings so the model can learn domain-specific token meaning. But the embeddings hold the most compressed form of the model's general language knowledge, so modifying them is what risks catastrophic forgetting. Two mechanisms keep this in check: LoRA and data mixing.


### LoRA

LoRA inserts small trainable matrices alongside the frozen base weights: the pretrained weights stay untouched and the adapters capture the domain-specific shift (Hu et al. 2021). Keeping the base frozen and restricting updates to low-rank adapters helps prevent catastrophic forgetting, because it limits how far the model's behavior can drift from its pretrained state. We run this as QLoRA (4-bit) for memory headroom on the Kaggle T4 GPU. The full memory argument for LoRA over full fine-tuning is detailed in Part 3.

Unlike standard SFT, where the embeddings stay frozen because the vocabulary context is unchanged, CPT also trains `embed_tokens` and `lm_head`, so the model can adapt the representation of domain-specific tokens to their E/E meaning (Han & Han 2024). CPT uses a high rank (here: r=64) to give the adapters enough capacity for this domain shift. At that rank, standard LoRA scaling (α/r) scales the updates down too aggressively, so rsLoRA is used instead. Scaling by α/√r keeps update magnitudes stable at high rank (Kalajdzievski 2023).


### Data Mixing

Restricting the weight updates to a low-rank subspace through LoRA largely prevents catastrophic forgetting, but training purely on domain text still risks degrading general language fluency. Mixing a small percentage of general-domain data alongside the automotive E/E domain chunks counteracts this (Ibrahim et al. 2024).


## Toy Setup

| Setting | Value |
|---|---|
| Base model | `unsloth/Llama-3.2-1B-bnb-4bit` (4-bit) |
| Platform | Kaggle T4 (15 GB VRAM) |
| LoRA | r=64, α=64, rsLoRA; targets: attention + FFN + `embed_tokens` + `lm_head` |
| Learning rate | 1e-4 (embeddings decoupled at 1e-5), cosine schedule, 6 warmup steps |
| Training | 2 epochs / 58 steps, effective batch 8 (2×4) |
| Data | 220 domain + 11 WikiText-2 (5%) = 231 train; 25 validation |
| Trainable params | ~316M / 1.81B (17.4%) |
| Checkpoint | epoch 1 (`load_best_model_at_end` on validation loss) |


## Results

#### Validation perplexity

| | Pre-CPT | Post-CPT (best, epoch 1) |
|---|---|---|
| Perplexity | 12.89 | 11.73 |

#### Training / validation loss

| Epoch | Train Loss | Val Loss |
|---|---|---|
| 1 | 2.403 | **2.462** |
| 2 | 1.959 | 2.469 |

Training loss falls over two epochs while validation loss is lowest after the first epoch and rises afterwards. On a corpus this small the model begins to overfit almost immediately, so the epoch-1 checkpoint is selected (`load_best_model_at_end` on validation loss).

#### Test prompt

```python
TEST_PROMPT = ["In CAN bus arbitration, the message that wins the bus is the one with"]
```
On the well-covered topic of CAN arbitration, the test prompt showed a qualitative improvement. The base model only restates that the highest-priority message wins without saying how priority is set, whereas after CPT the model gives the actual mechanism. It states that the lowest-ID message wins, using CAN's 11-bit identifier. It still shows toy-scale limits (an incorrect Ethernet analogy, repetition), but the domain-specific mechanism is now precise.


## Limitations

| Limitation | Impact | Production fix |
|---|---|---|
| Small data corpus (~430k tokens, ~245 chunks) | see part1 limitations | see part1 limitations |
| Hyperparameters not empirically tuned | Key choices (LoRA rank, learning rates, epoch count, data-mixing ratio) were taken from literature and Unsloth recommendations as starting points and have not been optimized. | Run controlled experiments over the hyperparameters and tune them to get the best validation loss / downstream eval. |
| 4-bit quantization (QLoRA) | Quantizing the base weights to 4-bit introduces rounding errors and can cap fidelity compared to 16-bit LoRA. | Train in 16-bit LoRA. |
| LoRA rank ceiling (r=64) | LoRA constrains the weight update to a rank-64 subspace, capping how much the model can change. | Increase the LoRA rank (up to full fine-tuning) for more adaptation capacity, as long as catastrophic forgetting stays controlled. |
| Evaluation by perplexity only | CPT is judged only by validation perplexity, which does not directly measure domain task performance. | Add task-level evaluation (Part 4). |


## References

- McCormick, C. (2025). *Continuing Pre-Training on Raw Text.* mccormickml.com/2025/01/18/continuing-pre-training-on-raw-text/.
- Gururangan, S. et al. (2020). *Don't Stop Pretraining: Adapt Language Models to Domains and Tasks.* ACL 2020. arXiv:2004.10964
- Zhou, C. et al. (2023). *LIMA: Less Is More for Alignment.* arXiv:2305.11206.
- Han, D. & Han, M. (2024). *Continued Pretraining with Unsloth.* unsloth.ai/blog/contpretraining.
- Hu, E. J. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685
- Kalajdzievski, D. (2023). *A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA.* arXiv:2312.03732
- Ibrahim, A. et al. (2024). *Simple and Scalable Strategies to Continually Pre-train Large Language Models.* arXiv:2403.08763.