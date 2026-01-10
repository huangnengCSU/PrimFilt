use clap::Parser;
use rust_htslib::{bam, bam::Read};
use std::time::Instant;
use std::collections::{HashMap, HashSet};

use rayon::prelude::*;
use std::sync::{Arc, Mutex};

mod util;
use util::load_reference;

mod logs;
use logs::init_logger;
use log::info;

mod annotation;
use annotation::{load_gene_introns_from_annotation, build_gene_tree, query_gene_tree};

use crate::internalpriming::read_bam;
use crate::util::Region;

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

    /// Discarded output BAM file
    #[arg(short='d', long)]
    discarded_output: Option<String>,

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
    let discarded_output_bam_file = arg.discarded_output.clone();
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
    info!("Filtered output BAM file: {}", output_bam_file);
    if let Some(discarded) = &discarded_output_bam_file {
        info!("Discarded output BAM file: {}", discarded);
    } else {
        info!("No discarded output BAM file.");
    }
    info!("Window size: {}", window_size);
    info!("Fraction of A's: {}", fraction);
    info!("Number of threads: {}", num_threads);

    let mut gene_introns: HashMap<String, HashMap<String, HashSet<String>>> = HashMap::new();
    let mut gene_regions = HashMap::new();
    if annotation_file.is_some() {
        (gene_introns, gene_regions) = load_gene_introns_from_annotation(annotation_file.unwrap().as_str());
    }

    let ref_seqs = load_reference(&reference_file);

    let mut bam: bam::Reader = bam::Reader::from_path(&input_bam_file).expect("Error opening BAM file");
    let header = bam::Header::from_template(bam.header());
    let fw: bam::Writer = bam::Writer::from_path(output_bam_file, &header, bam::Format::Bam).expect("Error opening output BAM file");
    let mutex_fw = Arc::new(Mutex::new(fw));
    let mutex_discarded_fw = if let Some(discarded_file) = discarded_output_bam_file {
        Some(Arc::new(Mutex::new(bam::Writer::from_path(discarded_file, &header, bam::Format::Bam).expect("Error opening discarded output BAM file"))))
    } else {
        None
    };

    let pool = rayon::ThreadPoolBuilder::new().num_threads(num_threads).build().unwrap();
    pool.install(|| {
        info!("Thread pool with {} threads initialized.", num_threads);
        ref_seqs.par_iter().for_each(|(chr, seq)| {
            let empty_gene_regions: HashMap<String, Region> = HashMap::new();   // in case no providing annotation
            let empty_gene_introns: HashMap<String, HashSet<String>> = HashMap::new();  // in case no providing annotation
            let chr_gene_regions= gene_regions.get(chr).unwrap_or(&empty_gene_regions);
            let chr_gene_introns = gene_introns.get(chr).unwrap_or(&empty_gene_introns);
            let chr_gene_tree = build_gene_tree(&chr_gene_regions);
            let (records, discarded_records) = read_bam(&input_bam_file, &chr_gene_tree, &chr_gene_introns, &ref_seqs, chr, window_size, fraction);
            info!("Number of records processed for chromosome {}: {}", chr, records.len());
            let mut writer = mutex_fw.lock().unwrap();
            for record in records {
                writer.write(&record).expect("Error writing BAM record");
            }
            if let Some(discarded_mutex) = &mutex_discarded_fw {
                let mut discarded_writer = discarded_mutex.lock().unwrap();
                for record in discarded_records {
                    discarded_writer.write(&record).expect("Error writing discarded BAM record");
                }
            }
        });
    });

    let duration = start.elapsed();
    info!("Processing completed in: {:?}", duration);
}