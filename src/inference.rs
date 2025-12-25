use crate::{config::ModelConfig, data, model::ReasoningModel};
use burn::{
    module::Module,
    record::{CompactRecorder, Recorder},
    tensor::{Int, Tensor, backend::Backend},
};
use std::io::{self, Write};
use tokenizers::Tokenizer;

pub struct ReasoningAgent<B: Backend> {
    model: ReasoningModel<B>,
    tokenizer: Tokenizer,
    device: B::Device,
}

impl<B: Backend> ReasoningAgent<B> {
    // Constructor 1: Load Trained Weights
    pub fn load(device: B::Device) -> Self {
        let config = ModelConfig::new();
        println!("> Loading Model Architecture...");
        let model_skeleton = ReasoningModel::new(&config, &device);

        println!("> Loading Trained Weights (cognito_model)...");
        // CompactRecorder automatically handles the file extension (.mpk or .bin)
        let record = CompactRecorder::new()
            .load("cognito_model".into(), &device)
            .expect("Could not find 'cognito_model' artifact. Did you run training?");

        let model = model_skeleton.load_record(record);
        let tokenizer = data::load_tokenizer();

        Self {
            model,
            tokenizer,
            device,
        }
    }

    pub fn chat(&self) {
        println!("\n--- Cognito Inference Engine (Type 'quit' to exit) ---");
        let mut history = String::new();

        loop {
            print!("\nUser: ");
            io::stdout().flush().unwrap();

            let mut input = String::new();
            io::stdin().read_line(&mut input).unwrap();
            let input = input.trim();

            if input == "quit" {
                break;
            }

            // Format input for the model
            // TinyStories is prose, so we just feed the prompt directly.
            // For a chat model, we'd use "<|user|>{input}<|assistant|>"
            let prompt = format!("{}{}", history, input);

            println!("Assistant (Generating):");
            let response = self.generate(&prompt, 50); // Gen 50 tokens

            // Simple history management (sliding window)
            history.push_str(&input);
            history.push_str(&response);
        }
    }

    fn generate(&self, prompt: &str, max_tokens: usize) -> String {
        // 1. Encode
        let encoding = self.tokenizer.encode(prompt, true).unwrap();
        let mut tokens = encoding.get_ids().to_vec();

        // 2. Generation Loop
        for _ in 0..max_tokens {
            let seq_len = tokens.len();

            // Context Window Safety Check
            if seq_len >= 1024 {
                break;
            }

            let input = Tensor::<B, 1, Int>::from_ints(tokens.as_slice(), &self.device)
                .reshape([1, seq_len]);

            let logits = self.model.forward(input);

            // Greedy Sampling (Top-1)
            let [batch, _, vocab] = logits.dims();
            let next_token_logits = logits
                .slice([0..1, seq_len - 1..seq_len])
                .reshape([batch, vocab]);

            let next_token_tensor = next_token_logits.argmax(1);
            let next_token_scalar =
                next_token_tensor.into_data().as_slice::<i32>().unwrap()[0] as u32;

            // Stream to console
            let decode_chunk = self.tokenizer.decode(&[next_token_scalar], true).unwrap();
            print!("{}", decode_chunk);
            io::stdout().flush().unwrap();

            tokens.push(next_token_scalar);

            // Stop tokens (End of Text or Newline often implies end of sentence in prose)
            if next_token_scalar == 50256 || decode_chunk.contains("\n") {
                break;
            }
        }
        println!(); // Newline at end

        // Return just the new part
        let full_text = self.tokenizer.decode(&tokens, true).unwrap();
        full_text.replace(prompt, "")
    }
}
