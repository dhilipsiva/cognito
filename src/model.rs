use crate::config::ModelConfig;
use burn::{
    nn::{
        Dropout,
        DropoutConfig,
        Embedding,
        EmbeddingConfig,
        Gelu, // <--- Add Gelu
        Linear,
        LinearConfig,
        RmsNorm,
        RmsNormConfig,
        attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig},
    },
    prelude::*,
    tensor::backend::Backend,
};

#[derive(Module, Debug)]
pub struct ReasoningBlock<B: Backend> {
    attention: MultiHeadAttention<B>,
    norm_1: RmsNorm<B>,

    // --- MLP Components ---
    mlp_fc1: Linear<B>, // Up-projection (d_model -> 4*d_model)
    mlp_act: Gelu,      // Non-linearity (The "Logic" happens here)
    mlp_fc2: Linear<B>, // Down-projection (4*d_model -> d_model)
    // ----------------------
    norm_2: RmsNorm<B>,
    dropout: Dropout,
}

impl<B: Backend> ReasoningBlock<B> {
    pub fn new(config: &ModelConfig, device: &B::Device) -> Self {
        let attention = MultiHeadAttentionConfig::new(config.d_model, config.num_heads)
            .with_dropout(config.dropout)
            .init(device);

        let norm_1 = RmsNormConfig::new(config.d_model)
            .with_epsilon(1e-5)
            .init(device);

        // Standard Transformer MLP: Expansion factor of 4
        let hidden_dim = config.d_model * 4;

        let mlp_fc1 = LinearConfig::new(config.d_model, hidden_dim).init(device);
        let mlp_act = Gelu::new();
        let mlp_fc2 = LinearConfig::new(hidden_dim, config.d_model).init(device);

        let norm_2 = RmsNormConfig::new(config.d_model)
            .with_epsilon(1e-5)
            .init(device);

        let dropout = DropoutConfig::new(config.dropout).init();

        Self {
            attention,
            norm_1,
            mlp_fc1, // Added
            mlp_act, // Added
            mlp_fc2, // Added
            norm_2,
            dropout,
        }
    }

    pub fn forward(&self, input: Tensor<B, 3>, mask: Tensor<B, 3, Bool>) -> Tensor<B, 3> {
        let x = input;

        // 1. Self-Attention (Causal)
        let norm_x = self.norm_1.forward(x.clone());
        let attn_out = self
            .attention
            .forward(MhaInput::new(norm_x.clone(), norm_x.clone(), norm_x.clone()).mask_attn(mask));
        let x = x + self.dropout.forward(attn_out.context);

        // 2. FeedForward (MLP) with Expansion + GELU
        let norm_x = self.norm_2.forward(x.clone());

        // Expansion -> Activation -> Compression
        let mlp_out = self.mlp_fc1.forward(norm_x);
        let mlp_out = self.mlp_act.forward(mlp_out);
        let mlp_out = self.mlp_fc2.forward(mlp_out);

        let x = x + self.dropout.forward(mlp_out);

        x
    }
}

// Keep ReasoningModel struct exactly as it was...
// (Copy the ReasoningModel struct and impl from the previous step here)
// ...
#[derive(Module, Debug)]
pub struct ReasoningModel<B: Backend> {
    token_embedding: Embedding<B>,
    pos_embedding: Embedding<B>,
    blocks: Vec<ReasoningBlock<B>>,
    norm: RmsNorm<B>,
    output: Linear<B>,
}

impl<B: Backend> ReasoningModel<B> {
    pub fn new(config: &ModelConfig, device: &B::Device) -> Self {
        let token_embedding = EmbeddingConfig::new(config.vocab_size, config.d_model).init(device);

        let pos_embedding = EmbeddingConfig::new(config.max_seq_len, config.d_model).init(device);

        let blocks = (0..config.num_layers)
            .map(|_| ReasoningBlock::new(config, device))
            .collect();

        let norm = RmsNormConfig::new(config.d_model)
            .with_epsilon(1e-5)
            .init(device);

        let output = LinearConfig::new(config.d_model, config.vocab_size).init(device);

        Self {
            token_embedding,
            pos_embedding,
            blocks,
            norm,
            output,
        }
    }

    pub fn forward(&self, item: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch_size, seq_len] = item.dims();
        let device = &item.device();

        let indices = Tensor::arange(0..seq_len as i64, device)
            .reshape([1, seq_len])
            .repeat_dim(0, batch_size);

        let mut x = self.token_embedding.forward(item) + self.pos_embedding.forward(indices);

        let mask = Tensor::<B, 2, Int>::ones([seq_len, seq_len], device)
            .tril(0)
            .bool()
            .unsqueeze::<3>()
            .expand([batch_size, seq_len, seq_len]);

        for block in &self.blocks {
            x = block.forward(x, mask.clone());
        }

        let x = self.norm.forward(x);
        self.output.forward(x)
    }
}
