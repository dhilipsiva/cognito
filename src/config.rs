use burn::config::Config;

#[derive(Config, Debug)]
pub struct ModelConfig {
    /// Number of attention heads.
    /// We use 16 heads for fine-grained focus.
    #[config(default = 16)]
    pub num_heads: usize,

    /// The model dimension (Width).
    /// 2048 is "Narrow" compared to Llama-70B, but dense.
    #[config(default = 2048)]
    pub d_model: usize,

    /// Number of layers (Depth).
    /// 24 layers provides sufficient depth for multi-step reasoning.
    #[config(default = 24)]
    pub num_layers: usize,

    /// The size of the vocabulary.
    /// We will use the standard GPT-4/Llama size (approx 50k-100k).
    #[config(default = 50257)]
    pub vocab_size: usize,

    /// Max sequence length (Context Window).
    /// 4096 tokens allow for long "Thought Traces".
    #[config(default = 4096)]
    pub max_seq_len: usize,

    /// Dropout rate for regularization.
    #[config(default = 0.1)]
    pub dropout: f64,
}
