use crate::model::ReasoningModel;
use burn::{
    prelude::*,
    tensor::{Int, Tensor, backend::Backend},
};
use std::io::{self, Write};
use tokenizers::Tokenizer; // Added for smooth text streaming

pub struct ReasoningAgent<B: Backend> {
    model: ReasoningModel<B>,
    tokenizer: Tokenizer,
    device: B::Device,
}

impl<B: Backend> ReasoningAgent<B> {
    pub fn new(model: ReasoningModel<B>, tokenizer: Tokenizer, device: B::Device) -> Self {
        Self {
            model,
            tokenizer,
            device,
        }
    }

    pub fn generate(&self, prompt: &str, max_tokens: usize) -> String {
        // 1. Encode Prompt
        let encoding = self.tokenizer.encode(prompt, true).unwrap();
        let mut tokens = encoding.get_ids().to_vec();

        println!("\n--- Generating (Autoregressive) ---");
        print!("{}", prompt);
        io::stdout().flush().unwrap(); // Force print immediately

        // 2. Generation Loop
        for _ in 0..max_tokens {
            let seq_len = tokens.len();

            // Create input tensor [1, Seq_Len]
            let input = Tensor::<B, 1, Int>::from_ints(tokens.as_slice(), &self.device)
                .reshape([1, seq_len]);

            // Forward Pass
            let logits = self.model.forward(input);

            // extract dimensions safely
            let [batch_size, _, vocab_size] = logits.dims();

            // FIX: Use reshape([1, vocab]) instead of squeeze.
            // This preserves the batch dimension even if it is 1.
            let next_token_logits = logits
                .slice([0..1, seq_len - 1..seq_len])
                .reshape([batch_size, vocab_size]);

            // 3. Greedy Sampling
            let next_token_tensor = next_token_logits.argmax(1); // [Batch, 1] (indices)

            // Extract the scalar ID
            let next_token_scalar =
                next_token_tensor.into_data().as_slice::<i32>().unwrap()[0] as u32;

            // 4. Decode & Print
            let decode_chunk = self.tokenizer.decode(&[next_token_scalar], true).unwrap();
            print!("{}", decode_chunk);
            io::stdout().flush().unwrap();

            // 5. Append & Continue
            tokens.push(next_token_scalar);

            // GPT-4 End of Text token is typically 100257 (or <|endoftext|>).
            // We'll break on a few common stop tokens just in case.
            if next_token_scalar == 50256 || next_token_scalar == 100257 {
                break;
            }
        }
        println!("\n-----------------------------------");

        self.tokenizer.decode(&tokens, true).unwrap()
    }
}
