mod config;
mod data;
mod inference;
mod model;
mod train; // <--- Add this

use burn::backend::Autodiff; // Wrapper for training
use burn_cuda::{Cuda, CudaDevice};
use clap::{Parser, Subcommand};
use inference::ReasoningAgent;

// CLI Definition
#[derive(Parser)]
#[command(name = "cognito")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train the model on dataset.txt
    Train,
    /// Run inference (chat)
    Interact,
}

fn main() {
    // 32MB Stack for Deep Transformer
    let handler = std::thread::Builder::new()
        .name("cognito-runtime".into())
        .stack_size(32 * 1024 * 1024)
        .spawn(|| {
            run();
        })
        .unwrap();

    handler.join().unwrap();
}

fn run() {
    let args = Cli::parse();
    let device = CudaDevice::new(0);

    match args.command {
        Commands::Train => {
            println!("--- Starting Training Pipeline ---");
            // Wrap Cuda backend in Autodiff for Backpropagation
            train::train::<Autodiff<Cuda>>(device);
        }
        Commands::Interact => {
            println!("--- Starting Inference Engine ---");
            // Load the saved model onto the GPU
            let agent = ReasoningAgent::<Autodiff<Cuda>>::load(device);
            // Enter the loop
            agent.chat();
        }
    }
}
