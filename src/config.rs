use burn::config::Config;

#[derive(Config, Debug)]
pub struct ModelConfig {
    #[config(default = 8)]
    pub num_heads: usize,

    #[config(default = 1024)]
    pub d_model: usize,

    #[config(default = 24)]
    pub num_layers: usize,

    // FIX: Increased from 50257 to 102400 to fit GPT-4 Tokenizer
    #[config(default = 102400)]
    pub vocab_size: usize,

    #[config(default = 1024)]
    pub max_seq_len: usize,

    #[config(default = 0.1)]
    pub dropout: f64,
}
