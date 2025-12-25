# Cogito

**A High-Density Reasoning Agent written in Pure Rust.**

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Stack](https://img.shields.io/badge/stack-Burn_Explosion-orange)
![License](https://img.shields.io/badge/license-MIT-blue)

## 1. The Manifesto
Current Large Language Models are "Stochastic Parrots"—high knowledge density, low reasoning density. They memorize the internet but struggle to deduce $C$ from $A \to B$.

**Cogito** is the inverse.
* **Knowledge Sparse:** It does not know who won the 1998 World Cup. It does not care.
* **Reasoning Dense:** It understands causality, logic, and decomposition.
* **Tool Native:** If it needs facts, it calls a tool. It does not hallucinate.

This is a **System 2** thinker implemented Completely in Rust.

## 2. Architecture

### The Stack

* **Language:** Rust (2024 edition)
* **Framework:** [Burn](https://github.com/tracel-ai/burn) (Deep Learning)
* **Backend:** `burn-cuda` (fallback)
* **Tokenizer:** HuggingFace `tokenizers` (Rust native bindings)

### The Model Design
We utilize a Decoder-only Transformer optimized for logical depth over width.

| Component | Specification | Rationale |
| :--- | :--- | :--- |
| **Parameters** | ~1.5B - 3B | Optimal size for single-GPU reasoning saturation. |
| **Attention** | Flash Attention v2 | Hardware acceleration for the RTX 5090. |
| **Positional** | RoPE (Rotary) | extrapolation for long "thinking" context windows. |
| **Activation** | SwiGLU | Higher logical capacity than GELU. |
| **Precision** | `bf16` | Native Blackwell tensor core utilization. |

## 3. The Cognitive Loop

Cogito does not answer immediately. It enters a reasoning loop enforced by special tokens.

1.  **Ingest:** User prompt is tokenized.
2.  **Deliberate (`<think>`):** The model generates internal monologue tokens to decompose the problem.
3.  **Action (`<call>`):** If data is missing, the model pauses generation and emits a structured tool call.
4.  **Intervention (Rust Runtime):** The binary executes the tool (HTTP, Calc, Sandbox) and injects the output via `<result>`.
5.  **Synthesis:** The model absorbs the result and formulates the final answer.

## 4. Hardware Requirements

Development is targeted at high-end consumer silicon.

* **GPU:** NVIDIA RTX 4090 / 5090 (24GB+ VRAM required for training).
* **CPU:** AVX-512 support recommended (Ryzen 9 7950X/9950X).
* **RAM:** 64GB+ System RAM.

## 5. Getting Started

### Prerequisites
Ensure CUDA Toolkit 12.x is installed.

### Build
```bash
# Clone the repository
git clone [https://github.com/your-username/cogito.git](https://github.com/your-username/cogito.git)
cd cogito

# Run the training pipeline (requires prepared dataset)
cargo run --release --bin train -- --config config/reasoning_1b.toml

# Run inference (CLI mode)
cargo run --release --bin interact
```

## 6. Dataset Strategy

We do not train on CommonCrawl. We train on:

* Synthetic Logic: Generated chain-of-thought traces.
* Code/Math: OpenWebMath, GSM8K (for structural logic).
* Tool Use Traces: Synthetic conversations demonstrating <call> / <result> syntax.

Built with Rust.
