# Part 2: Continued Pre-Training — Design Write-up

Continued Pre-Training (CPT) adapts a general base model to a target domain by continuing next-token training on in-domain text. The goal is to turn a generic model into one that has internalized automotive E/E patterns before any instruction tuning.


## Training Objective

CPT continues the model's original causal language-modeling objective *next-token prediction*:
The training objective for CPT is standard next-token prediction: given a sequence of tokens x₁, x₂, …, xₙ₋₁, the model predicts xₙ.

The loss is cross-entropy averaged over all token positions in the chunk. Every token in the corpus simultaneously serves as both input context and training target and no manual labeling is needed. This is called *self-supervised learning*.

The model shifts its probability distribution toward automotive E/E text by repeatedly predicting the next token in domain-specific sequences, with the cross-entropy loss backpropagated to update the weights.



## Model Choice

`Llama-3.2-1B` was chosen as the base model. Running on a Kaggle T4 GPU, memory efficiency is key and given its strong Unsloth support with availability as a pre-quantized 4-bit checkpoint, this model makes a good choice for the toy setup.

#### Base model
`Llama-3.2-1B` is available in both Base and Instruct variants. The Instruct variant has already been fine-tuned on instruction-following data.
CPT on raw domain text would degrade this instruction-following capability because the unstructured E/E documents as training data do not contain the prompt/response structure the model learned to expect (McCormick 2025). 

Therefore the pipeline is Base Model → CPT → SFT: first absorb domain knowledge from raw text, then teach instruction format on top.


## Why CPT Before SFT

Domain-Adaptive Pre-Training (DAPT) gains scale with domain distance from the pretraining corpus. Domains well-represented in web text benefit little, while structurally absent domains benefit substantially (Gururangan et al. 2020).

Regardless of the exact pretraining corpus, automotive E/E is a narrow engineering niche with its own terminology and specification formats that have no meaningful presence in general web text.

For example, without CPT, the model may be more likely to interpret `CAN` as the English modal verb or noun than as *Controller Area Network*.

Further, a small number of instruction pairs cannot build domain knowledge but only teach instruction format on top of existing knowledge, therefore CPT is needed (Zhou et al. 2023).


## Learning Rate Strategy

The learning rate strategy for CPT addresses different stability concerns.

#### Learning Rate Schedule
CPT begins from a well-optimized pretrained checkpoint, not from random initialization. A full learning rate at step 0 risks destroying attention patterns that took hundreds of billions of training tokens to develop. A short linear warmup phase brings the learning rate to its target value gradually, keeping early updates small enough to preserve the pretrained structure. After warmup, a cosine decay schedule maintains a high learning rate during the main training phase for broad domain pattern absorption, then decays smoothly toward convergence.

#### Decoupled Learning Rate
Finally, not all model components require the same degree of adaptation. Vocabulary embeddings encode the most compressed form of semantic knowledge from pretraining. Applying the same update magnitude as attention and FFN layers risks destabilizing this structure. A decoupled, smaller embedding learning rate allows domain-specific token meaning to shift gradually without destroying the general semantic prior (Han & Han 2024).


## Avoiding Catastrophic Forgetting

CPT must update the vocabulary embeddings so the model can learn domain-specific token meaning. But the embeddings hold the most compressed form of the model's general language knowledge, so modifying them is exactly what risks catastrophic forgetting. Two mechanisms keep this in check: LoRA and data mixing.


### LoRA

LoRA inserts small trainable matrices alongside the frozen base weights: the pretrained weights stay untouched and the adapters capture the domain-specific shift (Hu et al. 2021). Keeping the base frozen and restricting updates to low-rank adapters helps prevent catastrophic forgetting, because it limits how far the model's behavior can drift from its pretrained state. We run this as QLoRA (4-bit) for memory headroom on the Kaggle T4 GPU. The full memory argument for LoRA over full fine-tuning is detailed in Part 3.

Unlike standard SFT, where the embeddings stay frozen because the vocabulary context is unchanged, CPT also trains `embed_tokens` and `lm_head`, so the model can adapt the representation of domain-specific tokens to their E/E meaning (Han & Han 2024). CPT uses a high rank (r=64) to give the adapters enough capacity for this domain shift. At that rank, standard LoRA scaling (α/r) scales the updates down too aggressively, so rsLoRA is used instead. Scaling by α/√r keeps update magnitudes stable at high rank (Kalajdzievski 2023).


### Data Mixing

Restricting the weight updates to a low-rank subspace through LoRA largely prevents catastrophic forgetting, but training purely on domain text still risks degrading general language fluency. Mixing a small percentage of general-domain data alongside the automotive E/E domain chunks counteracts this (Ibrahim et al. 2024).


## Toy Setup

| Setting | Value |
|---|---|
| Base model | `unsloth/Llama-3.2-1B-bnb-4bit` (4-bit) |
| Platform | Kaggle T4 (15 GB VRAM) |
| LoRA | r=64, α=32, rsLoRA; targets: attention + FFN + `embed_tokens` + `lm_head` |
| Learning rate | 1e-4 (embeddings decoupled at 1e-5), cosine schedule, 6 warmup steps |
| Training | 3 epochs / 174 steps, effective batch 8 (4×2) |
| Data | 442 domain + 22 WikiText-2 (5%) = 464 train chunks; 50 validation |
| Trainable params | 316M / 1.24B (17.4%) — dominated by `embed_tokens` |
| Checkpoint | epoch 2 (`load_best_model_at_end` on validation loss) |


## Results

#### Validation perplexity

| | Pre-CPT | Post-CPT (epoch 2) |
|---|---|---|
| Perplexity | 10.27 | 8.89 |

#### Training / validation loss

| Epoch | Train Loss | Val Loss |
|---|---|---|
| 1 | 2.309 | 2.188 |
| 2 | 1.968 | **2.184** |
| 3 | 1.834 | 2.211 |

Training loss falls steadily while validation loss bottoms out at epoch 2 and rises at epoch 3 — the model begins overfitting, so the epoch-2 checkpoint is selected.

After CPT the model produces coherent domain-style text instead of repetition loops. A full evaluation and comparison with the base model and after SFT will be provided in Part 4.




## Limitations

| Limitation | Impact | Production fix |
|---|---|---|
| Small data corpus (~1M tokens, ~500 chunks) | see part1 limitations | see part1 limitations |
| Hyperparameters not empirically tuned | Key choices (LoRA rank, learning rates, epoch count, data-mixing ratio) were taken from literature and Unsloth recommendations as starting points and have not been optimized on this task. | At full scale, run controlled experiments over the hyperparameters and tune them to get the best validation loss / downstream eval. |
| 4-bit quantization (QLoRA) | Quantizing the base weights to 4-bit introduces rounding error and can cap fidelity compared to 16-bit LoRA. | Train in bf16 LoRA when GPU memory allows. |
| LoRA rank ceiling (r=64) | Adapters capture at most 64 orthogonal update directions per matrix; very complex domain shifts may not fully fit. | Raise the rank up to full fine-tuning, as long as forgetting stays controlled. |


## References

- Gururangan, S. et al. (2020). *Don't Stop Pretraining: Adapt Language Models to Domains and Tasks.* ACL 2020. arXiv:2004.10964
- Hu, E. J. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685
- Kalajdzievski, D. (2023). *A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA.* arXiv:2312.03732
- Han, D. & Han, M. (2024). *Continued Pretraining with Unsloth.* unsloth.ai/blog/contpretraining.
- McCormick, C. (2025). *Continuing Pre-Training on Raw Text.* mccormickml.com.
- Zhou, C. et al. (2023). *LIMA: Less Is More for Alignment.* arXiv:2305.11206.
- Ibrahim, A. et al. (2024). *Simple and Scalable Strategies to Continually Pre-train Large Language Models.* arXiv:2403.08763.