use burn::tensor::{Tensor, backend::Backend};
use burn_cuda::{Cuda, CudaDevice};

fn main() {
    println!("--- Cognito: System 2 Reasoning Kernel (Windows Native) ---");

    // 1. Initialize the Device
    // Device 0 is your primary GPU (RTX 5090)
    let device = CudaDevice::new(0);

    println!("> Backend Initialized: CUDA");
    println!("> Target Device: {:?}", device);

    // 2. Run the Kernel
    run_inference_test::<Cuda>(device);
}

fn run_inference_test<B: Backend>(device: B::Device) {
    println!("> Allocating tensors on VRAM...");

    // Create a 2x3 tensor
    let tensor_1: Tensor<B, 2> = Tensor::from_floats([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);

    let tensor_2: Tensor<B, 2> = Tensor::from_floats([[0.5, 0.5, 0.5], [1.0, 2.0, 1.0]], &device);

    println!("> Tensor 1 Shape: {:?}", tensor_1.shape());

    // Perform an operation (Addition) on the GPU
    let result = tensor_1 + tensor_2;

    println!("> Result (Tensor 1 + Tensor 2):");
    println!("{}", result);

    println!("\n> SUCCESS: RTX 5090 is online and accessible via Rust.");
}
