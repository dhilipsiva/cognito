use burn::{data::dataloader::batcher::Batcher, data::dataset::Dataset, prelude::*};
use std::{
    fs::File,
    io::{BufRead, BufReader},
};
use tokenizers::Tokenizer;

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
            .filter(|l| l.trim().len() > 50) // Filter short garbage
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

        // HARDCODED CONSTANTS
        let pad_id = 0; // We hijack 0 for padding
        let eos_id = 100257; // Standard GPT-4 End-of-Text

        let mut max_len_in_batch = 0;
        let valid_encodings: Vec<_> = encodings
            .iter()
            .filter(|e| e.get_ids().len() >= 2)
            .collect();

        // 1. Calculate Max Length (including space for EOS)
        for encoding in &valid_encodings {
            let len = encoding.get_ids().len() + 1; // +1 for EOS
            if len > max_len_in_batch {
                max_len_in_batch = len;
            }
        }
        max_len_in_batch = max_len_in_batch.min(self.max_seq_len);

        let mut all_inputs = Vec::new();
        let mut all_targets = Vec::new();

        for encoding in valid_encodings {
            let ids = encoding.get_ids();
            // Truncate if too long (leave room for EOS)
            let take_len = ids.len().min(max_len_in_batch - 1);
            let mut seq = ids[0..take_len].to_vec();

            // Append EOS token
            seq.push(eos_id); // [A, B, C, EOS]

            let current_len = seq.len();

            // Input: [A, B, C]
            // Target: [B, C, EOS]
            let input_vec: Vec<i32> = seq[0..current_len - 1].iter().map(|&x| x as i32).collect();
            let target_vec: Vec<i32> = seq[1..current_len].iter().map(|&x| x as i32).collect();

            let mut final_input = input_vec;
            let mut final_target = target_vec;

            // Pad with 0s
            let pad_len = max_len_in_batch - current_len; // -1 accounts for split
            if pad_len > 0 {
                final_input.extend(std::iter::repeat(pad_id).take(pad_len));
                final_target.extend(std::iter::repeat(pad_id).take(pad_len));
            }

            all_inputs.extend(final_input);
            all_targets.extend(final_target);
        }

        let seq_len = max_len_in_batch - 1;
        let batch_size = all_inputs.len() / seq_len;

        let inputs = Tensor::<B, 1, Int>::from_ints(all_inputs.as_slice(), device)
            .reshape([batch_size, seq_len]);

        let targets = Tensor::<B, 1, Int>::from_ints(all_targets.as_slice(), device)
            .reshape([batch_size, seq_len]);

        ReasoningBatch { inputs, targets }
    }
}

pub fn load_tokenizer() -> Tokenizer {
    println!("> Loading Tokenizer...");
    let tokenizer =
        Tokenizer::from_pretrained("Xenova/gpt-4", None).expect("Failed to load tokenizer.");
    // We don't need to add special tokens technically, we just use the IDs raw
    tokenizer
}
