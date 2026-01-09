use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};

use flate2::read::GzDecoder;

use crate::util::Region;

#[derive(Default, Clone, Debug)]
pub struct ParsedAnnotation {
    pub gene_regions: HashMap<String, Region>,
    // key is gene_id, value is the region of the gene, 1-based, start inclusive, end exclusive
    pub gene_names: HashMap<String, String>,
    // key is gene_id, value is gene name
    pub gene_strands: HashMap<String, String>,
    // key is gene_id, value is strand
    pub exon_regions: HashMap<String, HashMap<String, Vec<Region>>>,
    // key is gene_id, value is a map of transcript_id to a vector of exon regions
    pub intron_regions: HashMap<String, HashMap<String, Vec<Region>>>,
    // key is gene_id, value is a map of transcript_id to a vector of intron regions
    pub intron_profiles: HashMap<String, IntronProfile>,
    // key is gene_id, value is the IntronProfile
}

#[derive(Default, Clone, Debug)]
pub struct IntronProfile {
    pub gene_id: String,
    pub sorted_junctions: Vec<(u32, u32)>,
    // sorted junctions, each tuple is (start, end) of the intron
    pub isoform_junctions: HashMap<String, Vec<i32>>,
    // key is transcript_id, value is the absence or presence of the junction in the transcript refer to sorted_junctions
}

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

pub fn parse_annotation_file(annotation_file: &str, gene_types: &HashSet<String>, hgnc_filter: bool) -> ParsedAnnotation {
    let mut gene_regions = HashMap::new();
    let mut gene_names = HashMap::new();
    let mut gene_strands = HashMap::new();
    let mut exon_regions: HashMap<String, HashMap<String, Vec<Region>>> = HashMap::new();
    let mut intron_regions: HashMap<String, HashMap<String, Vec<Region>>> = HashMap::new();
    let mut intron_profiles = HashMap::new();

    if hgnc_filter {
        println!("Filtering genes not in HGNC.");
    }

    let file_type = if annotation_file.contains(".gff3") { "gff3" } else { "gtf" };
    let reader: Box<dyn BufRead> = if annotation_file.ends_with(".gz") {
        Box::new(BufReader::new(GzDecoder::new(File::open(annotation_file).unwrap())))
    } else {
        Box::new(BufReader::new(File::open(annotation_file).unwrap()))
    };

    let mut hgnc_id = ".".to_string();

    for line in reader.lines().flatten() {
        if line.starts_with('#') { continue; }
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() < 9 { continue; }

        let feature_type = parts[2];
        let attrs = parse_attributes(parts[8], file_type);

        if feature_type == "gene" {
            let gene_id = attrs.get("gene_id").unwrap_or(&".".to_string()).to_string();
            let gene_type = attrs.get("gene_type").or_else(|| attrs.get("gene_biotype"));
            let empty_string = String::new();
            let tag = attrs.get("tag").unwrap_or(&empty_string);
            let dot_string = ".".to_string();
            let gene_name = attrs.get("gene_name").unwrap_or(&dot_string).to_string();
            hgnc_id = attrs.get("hgnc_id").unwrap_or(&dot_string).to_string();
            if hgnc_filter && hgnc_id == "." { continue; }
            if gene_type.map_or(false, |t| gene_types.contains(t)) && !tag.contains("readthrough") {
                gene_regions.insert(gene_id.clone(), Region {
                    chr: parts[0].to_string(),
                    start: parts[3].parse().unwrap_or(0),   // 1-based, inclusive
                    end: parts[4].parse().unwrap_or(0) + 1, // 1-based, exclusive
                    max_coverage: None,
                    gene_id: None,
                });
                gene_names.insert(gene_id.clone(), gene_name);
                gene_strands.insert(gene_id.clone(), parts[6].to_string());
            }
        } else if feature_type == "exon" {
            if hgnc_filter && hgnc_id == "." { continue; }
            let gene_type = attrs.get("gene_type").or_else(|| attrs.get("gene_biotype"));
            let transcript_id = attrs.get("transcript_id").unwrap_or(&".".to_string()).to_string();
            let gene_id = attrs.get("gene_id").unwrap_or(&".".to_string()).to_string();
            let empty_string = String::new();
            let tag = attrs.get("tag").unwrap_or(&empty_string);
            if gene_type.map_or(false, |t| gene_types.contains(t)) && !tag.contains("readthrough") {
                exon_regions
                    .entry(gene_id.clone())
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
            }
        }
    }

    // Generate intron regions
    for (gene_id, transcripts) in &exon_regions {
        for (transcript_id, exons) in transcripts {
            if exons.len() <= 1 { continue; }
            let mut sorted_exons = exons.clone();
            sorted_exons.sort_by_key(|r| r.start);
            for i in 1..sorted_exons.len() {
                let intron_start = sorted_exons[i - 1].end; // 1-based, inclusive
                let intron_end = sorted_exons[i].start; // 1-based, exclusive
                if intron_start < intron_end {
                    intron_regions
                        .entry(gene_id.clone())
                        .or_default()
                        .entry(transcript_id.clone())
                        .or_default()
                        .push(Region {
                            chr: sorted_exons[i - 1].chr.clone(),
                            start: intron_start,
                            end: intron_end,
                            max_coverage: None,
                            gene_id: None,
                        });
                }
            }
        }
    }

    // Generate intron profiles as IsoQuant
    for (gene_id, introns) in &intron_regions {
        let mut junction_set: HashSet<(u32, u32)> = HashSet::new();
        for (transcript_id, intron_list) in introns {
            for intron in intron_list {
                let junction = (intron.start, intron.end);
                junction_set.insert(junction);
            }
        }

        // Convert to Vec and sort
        let mut sorted_junctions: Vec<(u32, u32)> = junction_set.into_iter().collect();
        sorted_junctions.sort_by(|a, b| a.cmp(b));

        // Create isoform junctions
        let mut isoform_junctions: HashMap<String, Vec<i32>> = HashMap::new();
        for (transcript_id, intron_list) in introns {
            let mut present_vec = vec![-1; sorted_junctions.len()];
            for intron in intron_list {
                if let Some(pos) = sorted_junctions.iter().position(|&j| j == (intron.start, intron.end)) {
                    present_vec[pos] = 1; // Mark presence
                }
            }
            isoform_junctions.insert(transcript_id.clone(), present_vec);
        }
        let intron_profile = IntronProfile {
            gene_id: gene_id.clone(),
            sorted_junctions: sorted_junctions,
            isoform_junctions: isoform_junctions,
        };
        intron_profiles.insert(gene_id.clone(), intron_profile);
    }

    ParsedAnnotation {
        gene_regions,
        gene_names,
        gene_strands,
        exon_regions,
        intron_regions,
        intron_profiles,
    }
}

pub fn load_junctions_from_annotation(annotation_file: &str) -> HashMap<String, Vec<String>> {
    let file_type = if annotation_file.contains(".gff3") { "gff3" } else { "gtf" };
    let reader: Box<dyn BufRead> = if annotation_file.ends_with(".gz") {
        Box::new(BufReader::new(GzDecoder::new(File::open(annotation_file).unwrap())))
    } else {
        Box::new(BufReader::new(File::open(annotation_file).unwrap()))
    };

    let mut gene_regions = HashMap::new();
    let mut exon_regions: HashMap<String, HashMap<String, Vec<Region>>> = HashMap::new();
    let mut introns:HashMap<String, Vec<String>> = HashMap::new();

    for line in reader.lines().flatten() {
        if line.starts_with('#') { continue; }
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() < 9 { continue; }

        let feature_type = parts[2];
        let attrs = parse_attributes(parts[8], file_type);

        if feature_type == "gene" {
            let dot_string = ".".to_string();
            let gene_name = attrs.get("gene_name").unwrap_or(&dot_string).to_string();
            gene_regions.insert(gene_name, Region {
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
                    introns.entry(format!("{}:{}-{}", chr, intron_start, intron_end - 1))
                        .or_default()
                        .push(gene_name.clone());
                }
            }
        }
    }
    introns // key is junction chr:start-end, 1-based, both inclusive. value is a list of gene names containing this junction
}