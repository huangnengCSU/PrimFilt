use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};

use flate2::read::GzDecoder;

use crate::util::Region;

use bio::data_structures::interval_tree::IntervalTree;

fn parse_attributes(attributes: &str, file_type: &str) -> HashMap<String, String> {
    let mut field_map = HashMap::new();
    for attr in attributes.trim_end_matches(';').split(';') {
        let attr = attr.trim();
        if attr.is_empty() { continue; }
        if file_type == "gff3" {
            if let Some((k, v)) = attr.split_once('=') {
                field_map.insert(k.to_string(), v.replace('"', ""));
            }
        } else {
            let parts: Vec<&str> = attr.split_whitespace().collect();
            if parts.len() >= 2 {
                let key = parts[0];
                let value = parts[1].trim_matches('"');
                field_map.entry(key.to_string())
                    .and_modify(|e| e.push_str(&format!(",{}", value)))
                    .or_insert_with(|| value.to_string());
            }
        }
    }
    field_map
}

pub fn load_gene_introns_from_annotation(annotation_file: &str) -> (HashMap<String, HashMap<String, HashSet<String>>>, HashMap<String, HashMap<String, Region>>, HashMap<String, HashMap<String, HashSet<i64>>>) {
    let file_type = if annotation_file.contains(".gff3") { "gff3" } else { "gtf" };
    let reader: Box<dyn BufRead> = if annotation_file.ends_with(".gz") {
        Box::new(BufReader::new(GzDecoder::new(File::open(annotation_file).unwrap())))
    } else {
        Box::new(BufReader::new(File::open(annotation_file).unwrap()))
    };

    let mut gene_regions = HashMap::new();
    let mut exon_regions: HashMap<String, HashMap<String, Vec<Region>>> = HashMap::new();
    let mut introns: HashMap<String, HashMap<String, HashSet<String>>> = HashMap::new(); // chrom -> gene_name -> set of introns (1-based, both inclusive)
    let mut transcript_ends: HashMap<String, HashMap<String, HashSet<i64>>> = HashMap::new(); // chrom -> gene_name -> set of transcript end positions

    for line in reader.lines().flatten() {
        if line.starts_with('#') { continue; }
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() < 9 { continue; }

        let chr = parts[0].to_string();
        let feature_type = parts[2];
        let attrs = parse_attributes(parts[8], file_type);

        if feature_type == "gene" {
            let dot_string = ".".to_string();
            let gene_name = attrs.get("gene_name").unwrap_or(&dot_string).to_string();
            if !gene_regions.contains_key(&chr) {
                gene_regions.insert(chr.clone(), HashMap::new());
            }

            gene_regions.get_mut(&chr).unwrap().insert(gene_name, Region {
                chr: parts[0].to_string(),
                start: parts[3].parse().unwrap_or(0),   // 1-based, inclusive
                end: parts[4].parse().unwrap_or(0) + 1, // 1-based, exclusive
                max_coverage: None,
                gene_id: None,
            });
        } else if feature_type == "exon" {
            let dot_string = ".".to_string();
            let gene_name = attrs.get("gene_name").unwrap_or(&dot_string).to_string();
            let transcript_id = attrs.get("transcript_id").unwrap_or(&".".to_string()).to_string();
            exon_regions
                .entry(gene_name)
                .or_default()
                .entry(transcript_id.clone())
                .or_default()
                .push(Region {
                    chr: parts[0].to_string(),
                    start: parts[3].parse().unwrap_or(0),
                    end: parts[4].parse().unwrap_or(0) + 1,
                    max_coverage: None,
                    gene_id: None,
                });
        } else if feature_type == "transcript" {
            let dot_string = ".".to_string();
            let gene_name = attrs.get("gene_name").unwrap_or(&dot_string).to_string();
            let chr = parts[0].to_string();
            let end1 = parts[3].parse().unwrap_or(0i64);
            let end2 = parts[4].parse().unwrap_or(0i64);
            transcript_ends
                .entry(chr.clone())
                .or_default()
                .entry(gene_name.clone())
                .or_default()
                .extend([end1, end2]);
        }
    }

    // Generate intron regions
    for (gene_name, transcripts) in &exon_regions {
        if gene_name == "." { continue; }
        for (transcript_id, exons) in transcripts {
            if exons.len() <= 1 { continue; }
            let mut sorted_exons = exons.clone();
            sorted_exons.sort_by_key(|r| r.start);
            for i in 1..sorted_exons.len() {
                let intron_start = sorted_exons[i - 1].end; // 1-based, inclusive
                let intron_end = sorted_exons[i].start; // 1-based, exclusive
                if intron_start < intron_end {
                    let chr = sorted_exons[i - 1].chr.clone();
                    if !introns.contains_key(&chr) {
                        introns.insert(chr.clone(), HashMap::new());
                    }
                    if !introns.get(&chr).unwrap().contains_key(gene_name) {
                        introns.get_mut(&chr).unwrap().insert(gene_name.clone(), HashSet::new());
                    }
                    introns.get_mut(&chr).unwrap().get_mut(gene_name).unwrap().insert(format!("{}:{}-{}", chr, intron_start, intron_end - 1)); // 1-based, both inclusive

                }
            }
        }
    }
    (introns, gene_regions, transcript_ends)
}


pub fn build_gene_tree(chr_gene_regions: &HashMap<String, Region>) -> IntervalTree<u32, String> {
    let mut tree = IntervalTree::new();
    for (gene_name, gene_region) in chr_gene_regions.iter() {
        tree.insert(gene_region.start..gene_region.end, gene_name.clone()); // interval is [start, end), 1-based
    }
    tree
}

pub fn query_gene_tree(tree: &IntervalTree<u32, String>, start: u32, end: u32) -> Vec<String> {
    // start: 1-based, inclusive
    // end: 1-based, exclusive
    let mut gene_names = Vec::new();
    for interval in tree.find(start..end) {
        gene_names.push(interval.data().clone());
    }
    gene_names
}