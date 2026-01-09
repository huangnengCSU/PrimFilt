use std::fs;
use std::path::Path;
use std::collections::HashMap;
use bio::io::fasta;

pub fn get_bam_file_size<P: AsRef<Path>>(bam_path: P) -> u64 {
    fs::metadata(&bam_path)
        .expect("Failed to get BAM file metadata")
        .len() // size in bytes
}

#[derive(Default, Clone, Debug)]
pub struct Region {
    pub(crate) chr: String,
    pub(crate) start: u32,
    // 1-based, inclusive
    pub(crate) end: u32,
    // 1-based, exclusive
    pub(crate) max_coverage: Option<u32>,
    // max coverage of this region
    pub(crate) gene_id: Option<String>,
    // if load annotation, this field will tell which gene this region covers. Multiple gene separated by comma
}

pub fn load_reference(ref_path: &str) -> HashMap<String, Vec<u8>> {
    let mut ref_seqs: HashMap<String, Vec<u8>> = HashMap::new();
    let reader = fasta::Reader::from_file(ref_path).unwrap();
    for r in reader.records() {
        let ref_record = r.unwrap();
        ref_seqs.insert(ref_record.id().to_string(), ref_record.seq().to_vec());
    }
    ref_seqs
}