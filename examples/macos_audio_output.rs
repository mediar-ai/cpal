use cpal::traits::StreamTrait;
use cpal::traits::{DeviceTrait, HostTrait};
use cpal::{host_from_id, BufferSize, HostId, SampleRate, StreamConfig};

fn main() {
    use std::process::Command;
    use std::time::Duration;

    // Create a dummy device and config
    let host = host_from_id(HostId::ScreenCaptureKit).unwrap();
    let device = host.default_input_device().unwrap();
    let config = StreamConfig {
        channels: 2,
        sample_rate: SampleRate(48000),
        buffer_size: BufferSize::Default,
    };

    // Run the function multiple times to increase chances of detecting leaks
    for _ in 0..1 {
        let stream = device
            .build_input_stream(&config, |_data: &[i8], _: &_| {}, |_err| {}, None)
            .expect("Failed to build stream");

        // Play and pause the stream
        stream.play().expect("Failed to play stream");
        std::thread::sleep(Duration::from_millis(10000));
        stream.pause().expect("Failed to pause stream");
    }

    // Get the current process ID
    let pid = std::process::id();

    // Run the 'leaks' command
    let output = Command::new("leaks")
        .args(&[pid.to_string()])
        .output()
        .expect("Failed to execute leaks command");

    // Check the output for leaks
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    println!("leaks stdout: { }", stdout);
    println!("leaks stderr: {}", stderr);
}
