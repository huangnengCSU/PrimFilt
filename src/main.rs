use clap::Parser;
use rust_htslib::{bam, bam::Read};
use std::time::Instant;
use std::collections::HashMap;

use rayon::prelude::*;
use std::sync::{Arc, Mutex};

mod util;
use util::load_reference;

mod logs;
use logs::init_logger;
use log::info;

mod annotation;
use annotation::{load_junctions_from_annotation};

use crate::internalpriming::read_bam;

mod internalpriming;

#[derive(Parser)]
struct Args {
    /// Input BAM file
    #[arg(short='i', long)]
    input: String,

    /// Input REF file
    #[arg(short='r', long)]
    reference: String,

    /// Input Annotation file, optional
    #[arg(short='a', long)]
    annotation: Option<String>,

    /// Filtered output BAM file
    #[arg(short='o', long)]
    output: String,

    /// Window size for check interal priming
    #[arg(short='w', long, default_value_t = 20)]
    window_size: usize,

    /// Fraction of A's in the window to consider as internal priming
    #[arg(short='f', long, default_value_t = 0.7)]
    fraction: f32,

    /// Number of threads to use
    #[arg(short='t', long, default_value_t = 1)]
    threads: usize,
}




fn main() {
    let start = Instant::now();
    init_logger();
    let arg = Args::parse();
    let input_bam_file = arg.input.clone();
    let reference_file = arg.reference.clone();
    let annotation_file = arg.annotation.clone();
    let output_bam_file = arg.output.clone();
    let window_size = arg.window_size;
    let fraction = arg.fraction;
    let num_threads = arg.threads;

    info!("Input BAM file: {}", input_bam_file);
    info!("Reference file: {}", reference_file);
    if let Some(ann) = &annotation_file {
        info!("Annotation file: {}", ann);
    } else {
        info!("No annotation file provided.");
    }
    info!("Output BAM file: {}", output_bam_file);
    info!("Window size: {}", window_size);
    info!("Fraction of A's: {}", fraction);
    info!("Number of threads: {}", num_threads);

    let mut anno_intron_map = HashMap::new(); // key is junction chr:start-end, 1-based, both inclusive. value is a list of gene names containing this junction
    if annotation_file.is_some() {
        anno_intron_map = load_junctions_from_annotation(annotation_file.unwrap().as_str());
    }

    let ref_seqs = load_reference(&reference_file);

    let mut bam: bam::Reader = bam::Reader::from_path(&input_bam_file).expect("Error opening BAM file");
    let header = bam::Header::from_template(bam.header());
    let fw: bam::Writer = bam::Writer::from_path(output_bam_file, &header, bam::Format::Bam).expect("Error opening output BAM file");
    let mutex_fw = Arc::new(Mutex::new(fw));

    let pool = rayon::ThreadPoolBuilder::new().num_threads(num_threads).build().unwrap();
    pool.install(|| {
        info!("Thread pool with {} threads initialized.", num_threads);
        ref_seqs.par_iter().for_each(|(chr, seq)| {
            let records = read_bam(&input_bam_file, &anno_intron_map, &ref_seqs, chr, window_size, fraction);
            info!("Number of records processed for chromosome {}: {}", chr, records.len());
            let mut writer = mutex_fw.lock().unwrap();
            for record in records {
                writer.write(&record).expect("Error writing BAM record");
            }
        });
    });

    // for chr in ref_seqs.keys() {
    //     // info!("Reference sequence loaded for chromosome: {}", chr);
    //     let records = read_bam(&input_bam_file, &anno_intron_map, &ref_seqs, chr, window_size, fraction, num_threads);
    //     info!("Number of records processed for chromosome {}: {}", chr, records.len());
    // }

    let duration = start.elapsed();
    info!("Processing completed in: {:?}", duration);
}