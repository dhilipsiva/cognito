mod config;
mod data;
mod model; // <--- Add this

use burn::module::Module;
use burn::tensor::backend::Backend;
use burn_cuda::{Cuda, CudaDevice};
use config::ModelConfig;
use model::ReasoningModel;

fn main() {
    println!("--- Cognito: System 2 Reasoning Kernel ---");

    // 1. Hardware
    let device = CudaDevice::new(0);

    // 2. Model
    let config = ModelConfig::new();
    println!("> Configuration Loaded: 1.2B Parameters (approx)");
    init_model::<Cuda>(&config, &device);

    // 3. Data Check
    println!("\n> Initializing Tokenizer...");
    let tokenizer = data::load_tokenizer();
    let text = "Logic is the beginning of wisdom.";
    let encoding = tokenizer.encode(text, true).unwrap();
    println!("> Tokenizer Check: '{}' -> {:?}", text, encoding.get_ids());
}

fn init_model<B: Backend>(config: &ModelConfig, device: &B::Device) {
    let model = ReasoningModel::<B>::new(config, device);
    let num_params = model.num_params();
    println!("> Model Successfully Created!");
    println!("> Total Parameters: {:.2} Billion", num_params as f64 / 1e9);
}
