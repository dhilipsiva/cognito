use crate::{config::ModelConfig, data, model::ReasoningModel};
use burn::{
    module::Module,
    record::{CompactRecorder, Recorder},
    tensor::{Distribution, Int, Tensor, backend::Backend},
};
use std::io::{self, Write};
use tokenizers::Tokenizer;

pub struct ReasoningAgent<B: Backend> {
    model: ReasoningModel<B>,
    tokenizer: Tokenizer,
    device: B::Device,
}

impl<B: Backend> ReasoningAgent<B> {
    pub fn load(device: B::Device) -> Self {
        let config = ModelConfig::new();
        println!("> Loading Model Architecture...");
        let model_skeleton = ReasoningModel::new(&config, &device);

        println!("> Loading Trained Weights...");
        let record = CompactRecorder::new()
            .load("cognito_model".into(), &device)
            .expect("Could not find 'cognito_model'. Run training first.");

        let model = model_skeleton.load_record(record);
        let tokenizer = data::load_tokenizer();

        Self {
            model,
            tokenizer,
            device,
        }
    }

    pub fn chat(&self) {
        println!("\n--- Cognito Inference Engine (Sampled) ---");
        let mut history = String::new();

        loop {
            print!("\nUser: ");
            io::stdout().flush().unwrap();

            let mut input = String::new();
            io::stdin().read_line(&mut input).unwrap();
            if input.trim() == "quit" {
                break;
            }

            let prompt = format!("{}{}", history, input.trim());
            println!("Assistant:");

            // Generate with Temperature 0.7 (Creative but focused)
            let response = self.generate(&prompt, 50, 0.7);

            history.push_str(input.trim());
            history.push_str(&response);
        }
    }

    fn generate(&self, prompt: &str, max_tokens: usize, temperature: f64) -> String {
        let encoding = self.tokenizer.encode(prompt, true).unwrap();
        let mut tokens = encoding.get_ids().to_vec();

        let penalty = 1.2; // Aggressive penalty (1.1 - 1.5 is standard)

        for _ in 0..max_tokens {
            let seq_len = tokens.len();
            if seq_len >= 1024 {
                break;
            }

            // 1. Forward Pass
            let input = Tensor::<B, 1, Int>::from_ints(tokens.as_slice(), &self.device)
                .reshape([1, seq_len]);
            let logits = self.model.forward(input);

            // 2. Get Logits for next token
            let [batch, _, vocab] = logits.dims();
            let next_token_logits = logits
                .slice([0..1, seq_len - 1..seq_len])
                .reshape([batch, vocab]);

            // --- REPETITION PENALTY LOGIC ---
            // We need to bring logits to CPU to modify them easily (or use scatter math on GPU)
            // For simplicity/speed in this demo, we'll use a hacky CPU roundtrip for the penalty
            // (In production, you'd implement a GPU kernel for this)
            let mut logits_vec = next_token_logits.into_data().to_vec::<f32>().unwrap();

            // Look at the last 64 tokens to penalize
            let start_window = if tokens.len() > 64 {
                tokens.len() - 64
            } else {
                0
            };
            let context_window = &tokens[start_window..];

            for &token_id in context_window {
                let id = token_id as usize;
                if id < logits_vec.len() {
                    // If logit is positive, divide. If negative, multiply.
                    // Simplified approach: Just subtract a constant value
                    logits_vec[id] = logits_vec[id] - penalty;
                }
            }

            // Put back on GPU
            let penalized_logits = Tensor::<B, 1>::from_floats(logits_vec.as_slice(), &self.device)
                .reshape([batch, vocab]);
            // -------------------------------

            // 3. Temperature
            let scaled_logits = penalized_logits / temperature;

            // 4. Sampling (Argmax with Noise)
            let probs = burn::tensor::activation::softmax(scaled_logits, 1);
            let noise = Tensor::<B, 2>::random(
                [batch, vocab],
                Distribution::Uniform(0.0, 1.0),
                &self.device,
            );

            let perturbed_logits = probs + (noise * 0.1);
            let next_token_tensor = perturbed_logits.argmax(1);
            let next_token_scalar =
                next_token_tensor.into_data().as_slice::<i32>().unwrap()[0] as u32;

            let decode_chunk = self
                .tokenizer
                .decode(&[next_token_scalar], true)
                .unwrap_or("".to_string());
            print!("{}", decode_chunk);
            io::stdout().flush().unwrap();

            tokens.push(next_token_scalar);

            // if next_token_scalar == 100257 || decode_chunk.contains("\n") {
            //     break;
            // }

            if next_token_scalar == 100257 {
                break;
            }
        }
        println!();
        self.tokenizer
            .decode(&tokens, true)
            .unwrap()
            .replace(prompt, "")
    }
}
