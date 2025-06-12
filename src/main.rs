use mimalloc::MiMalloc;
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use rand::Rng;
use rayon::prelude::*;
use std::env;
use std::time::Instant;

// Sequential matrix multiplication
fn matmul_flat_sequential(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut result = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += a[i * n + k] * b[k * n + j];
            }
            result[i * n + j] = sum;
        }
    }
    result
}

// Parallel matrix multiplication
fn matmul_flat_parallel(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut result = vec![0.0; n * n];
    result.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += a[i * n + k] * b[k * n + j];
            }
            row[j] = sum;
        }
    });
    result
}

// Generate a random NxN matrix
fn generate_matrix(n: usize) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    (0..n * n).map(|_| rng.gen::<f64>()).collect()
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let n: usize = args.get(1).and_then(|x| x.parse().ok()).unwrap_or(1000);
    println!("Matrix Size: {}x{}", n, n);

    let start_mem = get_current_memory();
    let a = generate_matrix(n);
    let b = generate_matrix(n);
    let after_alloc = get_current_memory();
    println!("Init Matrix Overhead : Mem = {:.2} MB", after_alloc - start_mem);

    // Sequential
    let start_seq = Instant::now();
    let _seq_result = matmul_flat_sequential(&a, &b, n);
    let duration_seq = start_seq.elapsed();
    let after_seq_compute = get_current_memory();
    println!("Sequential       : Time = {:>7.4}s, Peak Mem  : {:.2} MB", duration_seq.as_secs_f64(), after_seq_compute - start_mem);

    // Parallel
    let start_par = Instant::now();
    let _par_result = matmul_flat_parallel(&a, &b, n);
    let duration_par = start_par.elapsed();
    let after_par_compute = get_current_memory();
    println!("Parallel         : Time = {:>7.4}s, Peak Mem = {:>6.2} MB",
             duration_par.as_secs_f64(),
             after_par_compute - start_mem
    );
}

// Windows-only memory tracking via WinAPI GetProcessMemoryInfo
#[cfg(target_os = "windows")]
fn get_current_memory() -> f64 {
    use std::mem::MaybeUninit;

    #[repr(C)]
    struct PROCESS_MEMORY_COUNTERS {
        cb: u32,
        PageFaultCount: u32,
        PeakWorkingSetSize: usize,
        WorkingSetSize: usize,
        QuotaPeakPagedPoolUsage: usize,
        QuotaPagedPoolUsage: usize,
        QuotaPeakNonPagedPoolUsage: usize,
        QuotaNonPagedPoolUsage: usize,
        PagefileUsage: usize,
        PeakPagefileUsage: usize,
    }

    extern "system" {
        fn GetCurrentProcess() -> *mut core::ffi::c_void;
    }

    #[link(name = "Psapi")]
    extern "system" {
        fn GetProcessMemoryInfo(
            Process: *mut core::ffi::c_void,
            ppsmemCounters: *mut PROCESS_MEMORY_COUNTERS,
            cb: u32,
        ) -> i32;
    }

    unsafe {
        let handle = GetCurrentProcess();
        let mut counters = MaybeUninit::<PROCESS_MEMORY_COUNTERS>::uninit();
        let size = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
        let success = GetProcessMemoryInfo(handle, counters.as_mut_ptr(), size);

        if success != 0 {
            let counters = counters.assume_init();
            (counters.WorkingSetSize as f64) / (1024.0 * 1024.0)
        } else {
            0.0
        }
    }
}
