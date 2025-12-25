use crate::{
    config::ModelConfig,
    data::{ReasoningBatcher, TextFileDataset, load_tokenizer},
    model::ReasoningModel,
};
use burn::{
    config::Config,
    data::dataloader::DataLoaderBuilder,
    nn::loss::CrossEntropyLossConfig,
    optim::{AdamWConfig, GradientsParams, Optimizer},
    prelude::Module,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
};

#[derive(Config, Debug)]
pub struct TrainingConfig {
    #[config(default = 2)] // Reduced from 4
    pub batch_size: usize,

    #[config(default = 1)]
    pub num_epochs: usize,

    #[config(default = 1e-4)]
    pub learning_rate: f64,
}

pub fn train<B: AutodiffBackend>(device: B::Device) {
    let config_model = ModelConfig::new();
    let config_train = TrainingConfig::new();

    // 1. Init Model & Optimizer
    let mut model = ReasoningModel::<B>::new(&config_model, &device);

    let mut optim = AdamWConfig::new().with_weight_decay(1e-5).init();

    // 2. Load Data
    println!("> Loading Tokenizer & Data...");
    let tokenizer = load_tokenizer();
    let batcher = ReasoningBatcher::new(tokenizer.clone(), 128);
    let dataset = TextFileDataset::new("dataset.txt");

    let dataloader = DataLoaderBuilder::new(batcher)
        .batch_size(config_train.batch_size)
        .shuffle(42)
        .num_workers(1)
        .build(dataset);

    println!("> Starting Training Loop (Custom)...");

    // 3. The "Bare Metal" Loop
    for epoch in 1..=config_train.num_epochs {
        println!("--- Epoch {} ---", epoch);

        for (iteration, batch) in dataloader.iter().enumerate() {
            // A. Forward Pass
            let item = batch.inputs;
            let targets = batch.targets;

            let logits = model.forward(item);

            // FIX: Capture device BEFORE reshaping consumes 'logits'
            let device = logits.device();

            // B. Loss Calculation
            let [batch_size, seq_len, vocab_size] = logits.dims();
            let targets_flatten = targets.reshape([batch_size * seq_len]);
            let logits_flatten = logits.reshape([batch_size * seq_len, vocab_size]);

            let loss = CrossEntropyLossConfig::new()
                .with_pad_tokens(Some(vec![0]))
                .init(&device) // FIX: Use the captured device
                .forward(logits_flatten, targets_flatten);

            // Print Loss every 10 steps
            if iteration % 10 == 0 {
                let loss_scalar = loss.clone().into_scalar();
                println!(
                    "[Ep {}][Iter {}] Loss: {:.4}",
                    epoch, iteration, loss_scalar
                );
            }

            if iteration >= 10_000 {
                println!("> Reached 10,000 steps. Stopping early to save model.");
                break; // Breaks the inner loop
            }

            // C. Backward Pass
            let grads = loss.backward();

            // D. Optimize
            let grads_params = GradientsParams::from_grads(grads, &model);
            model = optim.step(config_train.learning_rate, model, grads_params);
        }

        println!("> Epoch {} complete.", epoch);
        model
            .save_file("cognito_model", &CompactRecorder::new())
            .expect("Failed to save trained model");
        break;
    }

    println!("> Training Complete.");
}
