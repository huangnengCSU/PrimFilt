use std::collections::HashMap;
use log::info;

use rust_htslib::bam;
use rust_htslib::bam::ext::BamRecordExtensions;
use rust_htslib::bam::{HeaderView, Reader, Record};
use rust_htslib::bam::Read;
use rust_htslib::bam::record::Aux;


pub fn read_bam(bam_path: &str, anno_introns: &HashMap<String, Vec<String>>, ref_seqs: &HashMap<String, Vec<u8>>, chr: &str, window_size: usize, fraction: f32) -> Vec<Record> {
    let mut out_records = Vec::new();
    let mut bam = bam::IndexedReader::from_path(bam_path).expect("Failed to open BAM file");
    // bam.set_threads(num_threads).expect("Failed to set BAM threads");
    let header = bam.header().to_owned();
    let tid = header.tid(chr.as_bytes()).expect("Chromosome not found");
    let chr_len = header.target_len(tid).unwrap();
    bam.fetch((chr.as_bytes(), 0, chr_len)).expect("Failed to fetch region");
    for r in bam.records() {
        let record = r.expect("Failed to read BAM record");
        // Process the record as needed
        let start = record.reference_start(); // 0-based, inclusive
        let end = record.reference_end(); // 0-based, exclusive
        let cigar = record.cigar();
        let begin_cigar = cigar.iter().next();
        let end_cigar = cigar.iter().last();
        let mut win1_start; // 0-based, inclusive
        let mut win1_end; // 0-based, exclusive
        if let Some(cg) = begin_cigar {
            let op = cg.char();
            let len = cg.len() as i64;
            if op == 'S' || op == 'H' {
                win1_end = start - len;
            } else {
                win1_end = start;
            }
            win1_start = win1_end - window_size as i64;
        } else {
            win1_end = start;
            win1_start = win1_end - window_size as i64;
        }
        if win1_start < 0 {
            win1_start = 0;
        }
        if win1_end < 0 {
            win1_end = 0;
        }
        let mut win2_start;
        let mut win2_end;
        if let Some(cg) = end_cigar {
            let op = cg.char();
            let len = cg.len() as i64;
            if op == 'S' || op == 'H' {
                win2_start = end + len;
            } else {
                win2_start = end;
            }
            win2_end = win2_start + window_size as i64;
        } else {
            win2_start = end;
            win2_end = win2_start + window_size as i64;
        }
        if win2_start > chr_len as i64 {
            win2_start = chr_len as i64;
        }
        if win2_end > chr_len as i64 {
            win2_end = chr_len as i64;
        }
        // TODO: exceed start or end position on the chromosome
        let win1_ref_seq = ref_seqs[chr][win1_start as usize..win1_end as usize].to_ascii_lowercase();
        let win2_ref_seq = ref_seqs[chr][win2_start as usize..win2_end as usize].to_ascii_lowercase();
        // Calculate fraction of A's in the windows
        let win1_a_count = win1_ref_seq.iter().filter(|&&c| c == 'a' as u8).count();
        let win2_a_count = win2_ref_seq.iter().filter(|&&c| c == 'a' as u8).count();
        let win1_a_fraction = win1_a_count as f32 / window_size as f32;
        let win2_a_fraction = win2_a_count as f32 / window_size as f32;
        // Calculate fractions of T's in the windows
        let win1_t_count = win1_ref_seq.iter().filter(|&&c| c == 't' as u8).count();
        let win2_t_count = win2_ref_seq.iter().filter(|&&c| c == 't' as u8).count();
        let win1_t_fraction = win1_t_count as f32 / window_size as f32;
        let win2_t_fraction = win2_t_count as f32 / window_size as f32;
        // if record.qname() == b"" {
        //     info!("{:?}\t{},{},{},{}", std::str::from_utf8(record.qname()), win1_a_fraction, win2_a_fraction, win1_t_fraction, win2_t_fraction);
        //     info!("igv region: {}:{}-{},{}:{}-{}", chr, win1_start + 1, win1_end, chr, win2_start + 1, win2_end);
        //     info!("ref pos (1-based, inclusive): {}\t{}", start + 1, end);
        //     info!("{:?}\t{:?}", std::str::from_utf8(&win1_ref_seq), std::str::from_utf8(&win2_ref_seq));
        //     info!("{}:{}, {}:{}", begin_cigar.unwrap().char(), begin_cigar.unwrap().len(), end_cigar.unwrap().char(), end_cigar.unwrap().len());
        // }
        if win1_a_fraction >= fraction || win1_t_fraction >= fraction || win2_a_fraction >= fraction || win2_t_fraction >= fraction {
            // info!("{:?}\t{},{},{},{}", std::str::from_utf8(record.qname()), win1_a_fraction, win2_a_fraction, win1_t_fraction, win2_t_fraction);
            continue;
        } else {
            // This read is not internally primed
            // Handle accordingly (e.g., keep, log, etc.)
            out_records.push(record);
        }
    }
    out_records
}