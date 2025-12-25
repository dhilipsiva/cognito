use burn::{data::dataloader::batcher::Batcher, prelude::*};
use tokenizers::{AddedToken, Tokenizer};

#[derive(Clone, Debug)]
pub struct ReasoningBatcher {
    tokenizer: Tokenizer,
    max_seq_len: usize,
}

impl ReasoningBatcher {
    pub fn new(tokenizer: Tokenizer, max_seq_len: usize) -> Self {
        Self {
            tokenizer,
            max_seq_len,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ReasoningBatch<B: Backend> {
    pub inputs: Tensor<B, 2, Int>,
    pub targets: Tensor<B, 2, Int>,
}

impl<B: Backend> Batcher<B, String, ReasoningBatch<B>> for ReasoningBatcher {
    fn batch(&self, items: Vec<String>, device: &B::Device) -> ReasoningBatch<B> {
        let encodings = self
            .tokenizer
            .encode_batch(items, true)
            .expect("Encoding failed");

        // Simplified Logic: Just take the first valid item
        let mut all_inputs = Vec::new();
        let mut all_targets = Vec::new();

        for encoding in encodings {
            let ids = encoding.get_ids();
            if ids.len() < 2 {
                continue;
            }
            let len = ids.len().min(self.max_seq_len + 1);

            all_inputs.extend(ids[0..len - 1].iter().map(|&i| i as i32));
            all_targets.extend(ids[1..len].iter().map(|&i| i as i32));
            break;
        }

        let seq_len = all_inputs.len();

        // FIX: Added '&' before device
        let inputs =
            Tensor::<B, 1, Int>::from_ints(all_inputs.as_slice(), device).reshape([1, seq_len]);

        let targets =
            Tensor::<B, 1, Int>::from_ints(all_targets.as_slice(), device).reshape([1, seq_len]);

        ReasoningBatch { inputs, targets }
    }
}

pub fn load_tokenizer() -> Tokenizer {
    println!("> Downloading Tokenizer (gpt-4)...");
    let mut tokenizer =
        Tokenizer::from_pretrained("Xenova/gpt-4", None).expect("Failed to load tokenizer.");

    let special_tokens = vec![
        AddedToken::from("<think>", true),
        AddedToken::from("</think>", true),
        AddedToken::from("<call>", true),
        AddedToken::from("</call>", true),
        AddedToken::from("<result>", true),
        AddedToken::from("</result>", true),
    ];

    tokenizer.add_special_tokens(&special_tokens);
    println!("> Special Tokens Injected.");
    tokenizer
}
