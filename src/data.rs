use burn::{data::dataloader::batcher::Batcher, data::dataset::Dataset, prelude::*};
use std::{
    fs::File,
    io::{BufRead, BufReader},
};
use tokenizers::{AddedToken, Tokenizer};

#[derive(Clone, Debug)]
pub struct TextFileDataset {
    lines: Vec<String>,
}

impl TextFileDataset {
    pub fn new(path: &str) -> Self {
        println!("> Loading dataset from: {}", path);
        let file = File::open(path).expect("Could not open dataset file");
        let reader = BufReader::new(file);

        let lines: Vec<String> = reader
            .lines()
            .map(|l| l.unwrap())
            .filter(|l| l.trim().len() > 20) // FIX: Filter out short/empty lines more aggressively
            .collect();

        println!("> Loaded {} samples.", lines.len());
        Self { lines }
    }
}

impl Dataset<String> for TextFileDataset {
    fn get(&self, index: usize) -> Option<String> {
        self.lines.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.lines.len()
    }
}

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

        // 1. Calculate Max Length safely
        let mut max_len_in_batch = 0;

        // Filter out bad encodings first
        let valid_encodings: Vec<_> = encodings
            .iter()
            .filter(|e| e.get_ids().len() >= 2) // Must have at least input + target
            .collect();

        if valid_encodings.is_empty() {
            // Panic with a clear message if the whole batch is garbage
            // (In production, you might return a dummy batch, but for now we want to know)
            panic!(
                "DataLoader encountered a batch with zero valid sequences! Dataset contains garbage."
            );
        }

        for encoding in &valid_encodings {
            let len = encoding.get_ids().len();
            if len > max_len_in_batch {
                max_len_in_batch = len;
            }
        }

        // Cap at absolute max
        max_len_in_batch = max_len_in_batch.min(self.max_seq_len + 1);

        let mut all_inputs = Vec::new();
        let mut all_targets = Vec::new();

        let pad_id = 0;

        for encoding in valid_encodings {
            let ids = encoding.get_ids();
            // We already filtered for len >= 2

            let current_len = ids.len().min(max_len_in_batch);

            let input_ids = &ids[0..current_len - 1];
            let target_ids = &ids[1..current_len];

            let mut input_vec = input_ids.iter().map(|&i| i as i32).collect::<Vec<_>>();
            let mut target_vec = target_ids.iter().map(|&i| i as i32).collect::<Vec<_>>();

            let pad_len = max_len_in_batch - current_len;
            if pad_len > 0 {
                input_vec.extend(std::iter::repeat(pad_id).take(pad_len));
                target_vec.extend(std::iter::repeat(pad_id).take(pad_len));
            }
            if let Some(&max_id) = input_vec.iter().max() {
                if max_id >= 102400 {
                    panic!("Token ID {} exceeds model vocab size 102400!", max_id);
                }
            }
            all_inputs.extend(input_vec);
            all_targets.extend(target_vec);
        }

        // 2. Safe Division
        // max_len_in_batch is guaranteed >= 2 (because of filter above), so -1 is safe.
        let seq_len = max_len_in_batch - 1;
        let batch_size = all_inputs.len() / seq_len;

        // 3. Create Tensors
        let inputs = Tensor::<B, 1, Int>::from_ints(all_inputs.as_slice(), device)
            .reshape([batch_size, seq_len]);

        let targets = Tensor::<B, 1, Int>::from_ints(all_targets.as_slice(), device)
            .reshape([batch_size, seq_len]);

        ReasoningBatch { inputs, targets }
    }
}

pub fn load_tokenizer() -> Tokenizer {
    println!("> Loading Tokenizer...");
    let mut tokenizer =
        Tokenizer::from_pretrained("Xenova/gpt-4", None).expect("Failed to load tokenizer.");

    if tokenizer.token_to_id("<pad>").is_none() {
        tokenizer.add_special_tokens(&[AddedToken::from("<pad>", true)]);
    }
    tokenizer
}
