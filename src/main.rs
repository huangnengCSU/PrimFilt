use clap::Parser;
use rust_htslib::{bam, bam::Read};
use std::time::Instant;
use std::collections::{HashMap, HashSet};

use rayon::prelude::*;
use std::sync::{Arc, Mutex};

use std::io::{BufWriter, Write};

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
#[command(
    name = env!("CARGO_PKG_NAME"),
    version = env!("CARGO_PKG_VERSION"),
    author = env!("CARGO_PKG_AUTHORS"),
    about = env!("CARGO_PKG_DESCRIPTION")
)]
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

    /// Discarded output BAM file, optional
    #[arg(short='d', long)]
    discarded_output: Option<String>,

    /// Output BED file for training data, optional
    #[arg(short='b', long)]
    output_bed: Option<String>,

    /// Feature length for internal priming site, default: 240
    #[arg(short='l', long, default_value_t = 240)]
    feature_length: usize,

    /// Primer sequences (e.g. oligo(dT)) are trimmed from reads, default: false
    #[arg(short='p', long, default_value_t = false)]
    primers_trimmed: bool,

    /// Window size for check interal priming
    #[arg(short='w', long, default_value_t = 20)]
    window_size: usize,

    /// Fraction of A's in the window to consider as internal priming, default: 0.7.
    /// Recommended: 0.7 for primer-trimmed reads, 0.6 for non-trimmed reads.
    #[arg(short='f', long, default_value_t = 0.7)]
    fraction: f32,

    /// Maximum distance to known annotated transcript end
    #[arg(short='e', long, default_value_t = 100)]
    end_distance: i64,

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
    let output_bed_file = arg.output_bed.clone();
    let feature_length = arg.feature_length;
    let primers_trimmed = arg.primers_trimmed;
    let window_size = arg.window_size;
    let fraction = arg.fraction;
    let end_distance = arg.end_distance;
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
    if let Some(bed) = &output_bed_file {
        info!("Output BED file: {}", bed);
    } else {
        info!("No output BED file.");
    }
    info!("Primers trimmed: {}", primers_trimmed);
    info!("Window size: {}", window_size);
    info!("Fraction of A's: {}", fraction);
    info!("Number of threads: {}", num_threads);

    let mut gene_introns: HashMap<String, HashMap<String, HashSet<String>>> = HashMap::new();
    let mut gene_regions = HashMap::new();
    let mut transcript_ends: HashMap<String, HashMap<String, HashSet<i64>>> = HashMap::new();
    if annotation_file.is_some() {
        (gene_introns, gene_regions, transcript_ends) = load_gene_introns_from_annotation(annotation_file.unwrap().as_str());
    }

    let ref_seqs = load_reference(&reference_file);

    let mut bam: bam::Reader = bam::Reader::from_path(&input_bam_file).expect("Error opening BAM file");
    let header = bam::Header::from_template(bam.header());
    let fw: bam::Writer = bam::Writer::from_path(output_bam_file, &header, bam::Format::Bam).expect("Error opening output BAM file");
    let mutex_fw = Arc::new(Mutex::new(fw));

    let mutex_discarded_fw = if let Some(discarded_file) = discarded_output_bam_file {
        Some(Arc::new(Mutex::new(
            bam::Writer::from_path(discarded_file, &header, bam::Format::Bam)
                .expect("Error opening discarded output BAM file")
        )))
    } else {
        None
    };

    let mutex_bed_writer = if let Some(bed_file) = &output_bed_file {
        use std::io::BufWriter;
        let file = std::fs::File::create(bed_file)
            .expect("Error creating BED file");
        Some(Arc::new(Mutex::new(BufWriter::new(file))))
    } else {
        None
    };

    let pool = rayon::ThreadPoolBuilder::new().num_threads(num_threads).build().unwrap();
    pool.install(|| {
        info!("Thread pool with {} threads initialized.", num_threads);
        ref_seqs.par_iter().for_each(|(chr, seq)| {
            let empty_gene_regions: HashMap<String, Region> = HashMap::new();
            let empty_gene_introns: HashMap<String, HashSet<String>> = HashMap::new();
            let empty_gene_transcript_ends: HashMap<String, HashSet<i64>> = HashMap::new();
            let chr_gene_regions = gene_regions.get(chr).unwrap_or(&empty_gene_regions);
            let chr_gene_introns = gene_introns.get(chr).unwrap_or(&empty_gene_introns);
            let chr_gene_transcript_ends = transcript_ends.get(chr).unwrap_or(&empty_gene_transcript_ends);
            let chr_gene_tree = build_gene_tree(&chr_gene_regions);
            let (records, discarded_records, features) = read_bam(
                &input_bam_file,
                &chr_gene_tree,
                &chr_gene_introns,
                &chr_gene_transcript_ends,
                &ref_seqs,
                chr,
                primers_trimmed,
                window_size,
                fraction,
                end_distance,
                feature_length
            );
            info!("Number of records processed for chromosome {}: {}", chr, records.len());

            {
                let mut writer = mutex_fw.lock().unwrap();
                for record in records {
                    writer.write(&record).expect("Error writing BAM record");
                }
            }


            if let Some(discarded_mutex) = &mutex_discarded_fw {
                let mut discarded_writer = discarded_mutex.lock().unwrap();
                for record in discarded_records {
                    discarded_writer.write(&record).expect("Error writing discarded BAM record");
                }
            }


            if let Some(bed_mutex) = &mutex_bed_writer {
                use std::io::Write;
                let mut bed_writer = bed_mutex.lock().unwrap();
                for feature in &features {
                    writeln!(
                        bed_writer,
                        "{}\t{}\t{}\t{}\t{}\t{}\t{}",
                        feature.chr,
                        feature.start,
                        feature.end,
                        feature.readname,
                        feature.label,
                        String::from_utf8_lossy(&feature.read_sequence),
                        String::from_utf8_lossy(&feature.reference_sequence)
                    ).expect("Error writing to BED file");
                }
            }
        });
    });

    if let Some(bed_mutex) = &mutex_bed_writer {
        let mut bed_writer = bed_mutex.lock().unwrap();
        bed_writer.flush().expect("Error flushing BED file");
    }

    let duration = start.elapsed();
    info!("Processing completed in: {:?}", duration);
}