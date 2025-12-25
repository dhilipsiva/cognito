mod config;
mod data;
mod inference;
mod model; // <--- Add this

use burn::module::Module;
use burn::tensor::backend::Backend;
use burn_cuda::{Cuda, CudaDevice};
use config::ModelConfig;
use inference::ReasoningAgent;
use model::ReasoningModel;

fn main() {
    println!("--- Cognito: System 2 Reasoning Kernel ---");

    let device = CudaDevice::new(0);
    let config = ModelConfig::new();

    // 1. Init Model
    println!("> Initializing Model...");
    let model = ReasoningModel::<Cuda>::new(&config, &device);

    // 2. Init Tokenizer (With Special Tokens)
    let tokenizer = data::load_tokenizer();

    // 3. Create Agent
    let agent = ReasoningAgent::new(model, tokenizer, device);

    // 4. Test Inference (Untrained)
    // It will output garbage, but it proves the loop works.
    agent.generate("User: What is 2+2?\nAssistant:", 20);
}
