use std::collections::{HashMap, HashSet};
use bio::data_structures::interval_tree::ArrayBackedIntervalTree;
use log::info;

use rust_htslib::bam;
use rust_htslib::bam::ext::BamRecordExtensions;
use rust_htslib::bam::{HeaderView, Reader, Record};
use rust_htslib::bam::Read;
use rust_htslib::bam::record::Aux;
use crate::annotation::query_gene_tree;

fn major_bases(seq: &[u8]) -> (Vec<u8>, usize) {
    if seq.is_empty() {
        return (Vec::new(), 0);
    }

    let mut cnt = [0usize; 256];
    for &b in seq {
        cnt[b as usize] += 1;
    }

    // 候选集合：你可以只用 A/C/G/T，或加上 N
    let candidates = [b'a', b'c', b'g', b't'];

    let max_count = candidates
        .iter()
        .map(|&b| cnt[b as usize])
        .max()
        .unwrap_or(0);

    let majors: Vec<u8> = candidates
        .iter()
        .copied()
        .filter(|&b| cnt[b as usize] == max_count && max_count > 0)
        .collect();
    (majors, max_count)
}

fn contains_poly_run(
    seq: &[u8],
    run_len: usize,
    max_mismatch: u32,
    target_base: u8, // b'A' or b'T'
) -> bool {
    if run_len == 0 || seq.len() < run_len {
        return false;
    }

    let target = target_base.to_ascii_uppercase();

    #[inline]
    fn is_target(b: u8, target: u8) -> bool {
        b.to_ascii_uppercase() == target
    }

    let mut bad: u32 = seq[..run_len]
        .iter()
        .map(|&b| (!is_target(b, target)) as u32)
        .sum();

    if bad <= max_mismatch {
        return true;
    }

    for i in run_len..seq.len() {
        bad -= (!is_target(seq[i - run_len], target)) as u32;
        bad += (!is_target(seq[i], target)) as u32;

        if bad <= max_mismatch {
            return true;
        }
    }

    false
}


fn closest_distance(sorted: &[i64], target: i64) -> i64 {
    if sorted.is_empty() {
        return i64::MAX;
    }

    let idx = sorted.partition_point(|&x| x < target);

    let dist = |a: i64| -> i64 {
        ((a as i128) - (target as i128)).abs() as i64
    };

    match idx {
        0 => dist(sorted[0]),
        i if i == sorted.len() => dist(sorted[sorted.len() - 1]),
        i => {
            let dl = dist(sorted[i - 1]);
            let dr = dist(sorted[i]);
            dl.min(dr)
        }
    }
}

pub struct IPFeature {
    pub reference_sequence: Vec<u8>,
    pub read_sequence: Vec<u8>,
    pub chr: String,
    pub start: i64, // 1-based, inclusive
    pub end: i64,   // 1-based, exclusive
    pub readname: String,
    pub label: u8, // 1 for internal priming, 0 for non-internal priming
}

pub fn read_bam(bam_path: &str,
                chr_gene_tree: &ArrayBackedIntervalTree<u32, String>,
                chr_gene_introns: &HashMap<String, HashSet<String>>,
                chr_gene_transcript_ends: &HashMap<String, HashSet<i64>>,
                ref_seqs: &HashMap<String, Vec<u8>>,
                chr: &str,
                primers_trimmed: bool,
                window_size: usize,
                fraction: f32,
                end_distance: i64,
                feature_length: usize) -> (Vec<Record>, Vec<Record>, Vec<IPFeature>) {
    let mut out_records = Vec::new();
    let mut discarded_records = Vec::new();
    let mut out_features = Vec::new();
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
        let mut win1_start = 0; // 0-based, inclusive
        let mut win1_end= 0; // 0-based, exclusive
        let mut win2_start= 0;
        let mut win2_end= 0;
        let mut read_win1_start = 0;
        let mut read_win1_end = 0;
        let mut read_win2_start = 0;
        let mut read_win2_end = 0;
        if primers_trimmed == true {
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
        } else {
            win1_start = record.reference_start();  // 0-based, inclusive
            win1_end = win1_start + window_size as i64; // 0-based, exclusive
            win2_end = record.reference_end();  // 0-based, exclusive
            win2_start = win2_end - window_size as i64; // 0-based, inclusive
            // adjust if windows exceed boundaries
            if win1_end > win2_end {
                win1_end = win2_end;
            }
            if win2_start < win1_start {
                win2_start = win1_start;
            }
        }

        let mut read_pos = 0;
        if let Some(cg) = begin_cigar {
            let op = cg.char();
            let len = cg.len();
            if op == 'S' {
                read_win1_start = len;
                read_win2_end = len;
                read_pos = len;
            } else if op == 'H' {
                read_win1_start = 0;
                read_win2_end = 0;
                read_pos = 0;
            } else {
                read_win1_start = 0;
                read_win2_end = 0;
                read_pos = 0;
            }
        }

        // read window length is 2 * window_size
        read_win1_end = read_win1_start + window_size as u32; // 0-based, exclusive
        if read_win1_start >= window_size as u32 {
            read_win1_start = read_win1_start - window_size as u32; // 0-based, inclusive
        } else {
            read_win1_start = 0;
        }

        // parse intron to see if the read has annotated intron
        let mut pos = record.reference_start() as u32;  // 0-based
        let mut win1_matched_A_count = 0;
        let mut win1_matched_T_count = 0;
        let mut win2_matched_A_count = 0;
        let mut win2_matched_T_count = 0;
        let mut read_introns: Vec<String> = Vec::new();
        for cg in cigar.iter() {
            let op = cg.char();
            let len = cg.len() as u32;
            if op == 'N' {
                let intron_start = pos;
                let intron_end = pos + len;
                let intron_key = format!("{}:{}-{}", chr, intron_start + 1, intron_end); // 1-based, both inclusive
                read_introns.push(intron_key);
            }
            if op == 'M' || op == '=' || op == 'X' {
                for i in 0..len {
                    if (((pos + i) as i64) >= win1_start) && (((pos + i) as i64) < win1_end) {
                        let ref_base = ref_seqs[chr][(pos + i) as usize];
                        let read_base = record.seq().as_bytes()[(read_pos + i) as usize];
                        if ref_base == b'A' || ref_base == b'a' {
                            if read_base == b'A' || read_base == b'a' {
                                win1_matched_A_count += 1;
                            }
                        }
                        if ref_base == b'T' || ref_base == b't' {
                            if read_base == b'T' || read_base == b't' {
                                win1_matched_T_count += 1;
                            }
                        }
                    }

                    if (((pos + i) as i64) >= win2_start) && (((pos + i) as i64) < win2_end) {
                        let ref_base = ref_seqs[chr][(pos + i) as usize];
                        let read_base = record.seq().as_bytes()[(read_pos + i) as usize];
                        if ref_base == b'A' || ref_base == b'a' {
                            if read_base == b'A' || read_base == b'a' {
                                win2_matched_A_count += 1;
                            }
                        }
                        if ref_base == b'T' || ref_base == b't' {
                            if read_base == b'T' || read_base == b't' {
                                win2_matched_T_count += 1;
                            }
                        }
                    }
                }
                pos += len;
                read_pos += len;
                read_win2_end += len;
            }
            if op == 'D' {
                pos += len;
            }
            if op == 'I' {
                read_pos += len;
                read_win2_end += len;
            }
            if op == 'N' {
                pos += len;
            }
        }

        // read window length is 2 * window_size
        read_win2_start = read_win2_end - window_size as u32; // 0-based, inclusive
        if read_win2_end + (window_size as u32) < record.seq().len() as u32 {
            read_win2_end = read_win2_end + window_size as u32; // 0-based, exclusive
        } else {
            read_win2_end = record.seq().len() as u32;
        }


        let overlapping_genes: Vec<String> = query_gene_tree(&chr_gene_tree, start as u32 + 1, end as u32 + 1); // 1-based, start inclusive, end exclusive
        // let mut has_annotated_intron = false;
        // let mut annotated_intron = String::new();
        // for gene_name in overlapping_genes.iter() {
        //     if !chr_gene_introns.contains_key(gene_name) {
        //         continue;
        //     }
        //     let gene_introns = &chr_gene_introns[gene_name];
        //     for intron in read_introns.iter() {
        //         if gene_introns.contains(intron) {
        //             has_annotated_intron = true;
        //             annotated_intron = intron.clone();
        //             break;
        //         }
        //     }
        //     if has_annotated_intron {
        //         break;
        //     }
        // }

        // check if the read end is close to transcript ends
        let mut ends: HashSet<i64> = HashSet::new();
        for gene_name in overlapping_genes.iter() {
            if !chr_gene_transcript_ends.contains_key(gene_name) {
                continue;
            }
            let gene_transcript_ends = &chr_gene_transcript_ends[gene_name];
            ends.extend(gene_transcript_ends);
        }
        // sort ends
        let mut sorted_ends: Vec<i64> = ends.into_iter().collect();
        sorted_ends.sort_unstable();


        // ===========================================================================
        // 头部 240bp alignment (120bp in the aligned part and 120bp in the clipped part)
        // ===========================================================================
        let half_len = (feature_length / 2) as usize;
        let head_ref_pos = record.reference_start();

        // 获取读序列起始位置（跳过soft-clip）
        let head_read_pos = cigar.first()
            .filter(|op| op.char() == 'S')
            .map(|op| op.len() as usize)
            .unwrap_or(0);

        // ---------------------------
        // 提取参考序列窗口
        // ---------------------------
        let chr_seq = &ref_seqs[chr];
        let chr_len = chr_seq.len() as i64;

        let head_ref_seq: String = (head_ref_pos - half_len as i64..head_ref_pos + half_len as i64)
            .map(|p| {
                if p < 0 || p >= chr_len {
                    '*'
                } else {
                    (chr_seq[p as usize] as char).to_ascii_lowercase()
                }
            })
            .collect();

        // ---------------------------
        // 提取读序列左半部分（带padding）
        // ---------------------------
        let seq_bytes = record.seq().as_bytes();
        let read_len = seq_bytes.len() as i64;

        let head_read_seq_half1: String = (head_read_pos as i64 - half_len as i64..head_read_pos as i64)
            .map(|p| {
                if p < 0 || p >= read_len {
                    '*'
                } else {
                    (record.seq()[p as usize] as char).to_ascii_lowercase()
                }
            })
            .collect();

        // ---------------------------
        // 提取读序列右半部分（根据CIGAR，带padding）
        // ---------------------------
        let mut head_read_seq_half2 = String::with_capacity(half_len);
        let mut ref_pos = head_ref_pos;
        let mut read_pos = head_read_pos;
        let mut ref_idx = half_len; // 对应 head_ref_seq 中的索引位置

        for cg in cigar.iter() {
            if head_read_seq_half2.len() >= half_len {
                break; // 已经提取足够长度
            }

            match cg.char() {
                'M' | '=' | 'X' => {
                    let len = (cg.len() as usize).min(half_len - head_read_seq_half2.len());
                    for i in 0..len {
                        let read_base = record.seq()[read_pos + i];
                        head_read_seq_half2.push((read_base as char).to_ascii_lowercase());

                        // 验证参考序列匹配
                        debug_assert_eq!(
                            ref_seqs[chr][(ref_pos + i as i64) as usize],
                            head_ref_seq.as_bytes()[ref_idx]
                        );
                        ref_idx += 1;
                    }
                    ref_pos += len as i64;
                    read_pos += len;
                }
                'D' => {
                    let len = (cg.len() as usize).min(half_len - head_read_seq_half2.len());
                    for i in 0..len {
                        head_read_seq_half2.push('-');
                        // 验证参考序列匹配
                        debug_assert_eq!(
                            ref_seqs[chr][(ref_pos + i as i64) as usize],
                            head_ref_seq.as_bytes()[ref_idx]
                        );
                        ref_idx += 1;
                    }
                    ref_pos += len as i64;
                }
                'N' => {
                    let len = (cg.len() as usize).min(half_len - head_read_seq_half2.len());
                    for i in 0..len {
                        head_read_seq_half2.push('N');
                        // 验证参考序列匹配
                        debug_assert_eq!(
                            ref_seqs[chr][(ref_pos + i as i64) as usize],
                            head_ref_seq.as_bytes()[ref_idx]
                        );
                        ref_idx += 1;
                    }
                    ref_pos += len as i64;
                }
                'I' => {
                    read_pos += cg.len() as usize;
                }
                'S' | 'H' => {
                    // Soft/Hard clip - 跳过
                }
                _ => {}
            }
        }

        // 如果不够长度，用 '*' 填充
        while head_read_seq_half2.len() < half_len {
            head_read_seq_half2.push('*');
        }

        // 合并读序列两半
        let head_read_seq = format!("{}{}", head_read_seq_half1, head_read_seq_half2);

        let mut head_feature = IPFeature {
            reference_sequence: head_ref_seq.into_bytes(),
            read_sequence: head_read_seq.into_bytes(),
            chr: chr.to_string(),
            start: head_ref_pos - half_len as i64 + 1, // 转为1-based
            end: head_ref_pos + half_len as i64, // 1-based, exclusive
            readname: String::from_utf8_lossy(record.qname()).to_string(),
            label: 0, // 先标记为非内部引物，后续根据窗口特征更新
        };


        // ===========================================================================
        // 尾部 240bp alignment (120bp in the aligned part and 120bp in the clipped part)
        // ===========================================================================

        // 计算比对结束位置
        let mut tail_ref_pos = record.reference_start();
        let mut tail_read_pos = head_read_pos; // 从首个比对碱基开始

        for cg in cigar.iter() {
            match cg.char() {
                'M' | '=' | 'X' | 'D' => {
                    tail_ref_pos += cg.len() as i64;
                }
                _ => {}
            }
            match cg.char() {
                'M' | '=' | 'X' | 'I' => {
                    tail_read_pos += cg.len() as usize;
                }
                _ => {}
            }
        }

        // ---------------------------
        // 提取尾部参考序列窗口
        // ---------------------------
        let tail_ref_seq: String = (tail_ref_pos - half_len as i64..tail_ref_pos + half_len as i64)
            .map(|p| {
                if p < 0 || p >= chr_len {
                    '*'
                } else {
                    (chr_seq[p as usize] as char).to_ascii_lowercase()
                }
            })
            .collect();

        // ---------------------------
        // 提取尾部读序列左半部分（根据CIGAR反向，带padding）
        // ---------------------------
        let mut tail_read_seq_half1 = String::with_capacity(half_len);
        let mut ref_pos = tail_ref_pos - 1;
        let mut read_pos = tail_read_pos - 1;
        let mut ref_idx = half_len - 1; // 从中间往左

        // 反向遍历 CIGAR
        for cg in cigar.iter().rev() {
            if tail_read_seq_half1.len() >= half_len {
                break;
            }

            match cg.char() {
                'M' | '=' | 'X' => {
                    let len = (cg.len() as usize).min(half_len - tail_read_seq_half1.len());
                    for _ in 0..len {
                        let read_base = record.seq()[read_pos];
                        tail_read_seq_half1.push((read_base as char).to_ascii_lowercase());

                        debug_assert_eq!(
                            ref_seqs[chr][ref_pos as usize],
                            tail_ref_seq.as_bytes()[ref_idx]
                        );
                        ref_idx = ref_idx.wrapping_sub(1);
                        ref_pos -= 1;
                        read_pos -= 1;
                    }
                }
                'D' => {
                    let len = (cg.len() as usize).min(half_len - tail_read_seq_half1.len());
                    for _ in 0..len {
                        tail_read_seq_half1.push('-');
                        debug_assert_eq!(
                            ref_seqs[chr][ref_pos as usize],
                            tail_ref_seq.as_bytes()[ref_idx]
                        );
                        ref_idx = ref_idx.wrapping_sub(1);
                        ref_pos -= 1;
                    }
                }
                'N' => {
                    let len = (cg.len() as usize).min(half_len - tail_read_seq_half1.len());
                    for _ in 0..len {
                        tail_read_seq_half1.push('N');
                        debug_assert_eq!(
                            ref_seqs[chr][ref_pos as usize],
                            tail_ref_seq.as_bytes()[ref_idx]
                        );
                        ref_idx = ref_idx.wrapping_sub(1);
                        ref_pos -= 1;
                    }
                }
                'I' => {
                    read_pos -= cg.len() as usize;
                }
                'S' | 'H' => {
                    // Soft/Hard clip - 跳过
                }
                _ => {}
            }
        }

        // 填充不足部分
        while tail_read_seq_half1.len() < half_len {
            tail_read_seq_half1.push('*');
        }

        // 反向字符串（因为是从后往前收集的）
        let tail_read_seq_half1: String = tail_read_seq_half1.chars().rev().collect();

        // ---------------------------
        // 提取尾部读序列右半部分（soft-clip后的序列，带padding）
        // ---------------------------
        let tail_read_seq_half2: String = (tail_read_pos as i64..tail_read_pos as i64 + half_len as i64)
            .map(|p| {
                if p < 0 || p >= read_len {
                    '*'
                } else {
                    (record.seq()[p as usize] as char).to_ascii_lowercase()
                }
            })
            .collect();

        // 合并尾部读序列两半
        let tail_read_seq = format!("{}{}", tail_read_seq_half1, tail_read_seq_half2);

        let mut tail_feature = IPFeature {
            reference_sequence: tail_ref_seq.into_bytes(),
            read_sequence: tail_read_seq.into_bytes(),
            chr: chr.to_string(),
            start: tail_ref_pos - half_len as i64 + 1, // 转为1-based
            end: tail_ref_pos + half_len as i64, // 1-based, exclusive
            readname: String::from_utf8_lossy(record.qname()).to_string(),
            label: 0, // 先标记为非内部引物，后续根据窗口特征更新
        };


        if primers_trimmed {
            // primers trimmed
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

            // find the closest transcript end to win1_end
            let win1_dist = closest_distance(&sorted_ends, win1_end + 1); // win1_end is 0-base, exclusive, sorted_ends is 1-based
            // find the closest transcript end to win2_start
            let win2_dist = closest_distance(&sorted_ends, win2_start); // win2_start is 0-base, inclusive, sorted_ends is 1-based

            // if record.qname() == b"<read-name>" {
            //     info!("{:?}\t{},{},{},{}", std::str::from_utf8(record.qname()), win1_a_fraction, win1_t_fraction, win2_a_fraction, win2_t_fraction);
            //     info!("igv region: {}:{}-{},{}:{}-{}", chr, win1_start + 1, win1_end, chr, win2_start + 1, win2_end);
            //     info!("ref pos (1-based, inclusive): {}\t{}", start + 1, end);
            //     info!("{:?}\t{:?}", std::str::from_utf8(&win1_ref_seq), std::str::from_utf8(&win2_ref_seq));
            //     info!("{}:{}, {}:{}", begin_cigar.unwrap().char(), begin_cigar.unwrap().len(), end_cigar.unwrap().char(), end_cigar.unwrap().len());
            //     // info!("Annotated intron: {}", annotated_intron);
            //     if (win1_a_fraction >= fraction || win1_t_fraction >= fraction) && win1_dist > end_distance {
            //         info!("discarded by win1");
            //     } else if (win2_a_fraction >= fraction || win2_t_fraction >= fraction) && win2_dist > end_distance {
            //         info!("discarded by win2");
            //     } else {
            //         info!("not discarded");
            //     }
            // }

            if (win1_a_fraction >= fraction || win1_t_fraction >= fraction) && win1_dist > end_distance {
                discarded_records.push(record);
            } else if (win2_a_fraction >= fraction || win2_t_fraction >= fraction) && win2_dist > end_distance {
                discarded_records.push(record);
            } else {
                out_records.push(record);
            }

            if (win1_a_fraction >= fraction || win1_t_fraction >= fraction) && win1_dist > end_distance {
                head_feature.label = 1;
            }
            if (win2_a_fraction >= fraction || win2_t_fraction >= fraction) && win2_dist > end_distance {
                tail_feature.label = 1;
            }

            // if (win1_a_fraction >= fraction || win1_t_fraction >= fraction) && (!has_annotated_intron || win1_dist > end_distance) {
            //     discarded_records.push(record);
            // } else if (win2_a_fraction >= fraction || win2_t_fraction >= fraction) && (!has_annotated_intron || win2_dist > end_distance) {
            //     discarded_records.push(record);
            // } else {
            //     out_records.push(record);
            // }
        } else {
            // primers not trimmed
            let win1_ref_seq = ref_seqs[chr][win1_start as usize..win1_end as usize].to_ascii_lowercase(); // window-sized reference sequence within first mapped (Cigar M/X/=) part
            let win2_ref_seq = ref_seqs[chr][win2_start as usize..win2_end as usize].to_ascii_lowercase(); // window-sized reference sequence within last mapped (Cigar M/X/=) part
            let win1_read_seq = record.seq().as_bytes()[read_win1_start as usize..read_win1_end as usize].to_ascii_lowercase(); // 2 window-sized read sequence, half in clipped part and half in mapped part
            let (major_bases_win1,major_count_win1) = major_bases(&win1_read_seq);
            let win2_read_seq = record.seq().as_bytes()[read_win2_start as usize..read_win2_end as usize].to_ascii_lowercase(); // 2 window-sized read sequence, half in clipped part and half in mapped part
            let (major_bases_win2,major_count_win2) = major_bases(&win2_read_seq);
            let win1_a_fraction =  win1_matched_A_count as f32 / window_size as f32;
            let win1_t_fraction =  win1_matched_T_count as f32 / window_size as f32;
            let win2_a_fraction =  win2_matched_A_count as f32 / window_size as f32;
            let win2_t_fraction =  win2_matched_T_count as f32 / window_size as f32;

            // polyT occurs left side (win1) and polyA occurs right side (win2), otherwise not considered as internal priming
            let is_polyT_win1 = contains_poly_run(&win1_read_seq, 10, 1, b'T');
            let is_polyA_win2 = contains_poly_run(&win2_read_seq, 10, 1, b'A');

            // find the closest transcript end to win1_start
            let win1_dist = closest_distance(&sorted_ends, win1_start + 1); // win1_start is 0-base, inclusive, sorted_ends is 1-based
            // find the closest transcript end to win2_end
            let win2_dist = closest_distance(&sorted_ends, win2_end); // win2_start is 0-base, exclusive, sorted_ends is 1-based

            // if record.qname() == b"<read-name>" {
            //     info!("win1 polyT: {}", is_polyT_win1);
            //     info!("win2 polyA: {}", is_polyA_win2);
            //     info!("win1 dist: {}, win1_start: {}", win1_dist, win1_start + 1);
            //     info!("win2 dist: {}, win2_end: {}", win2_dist, win2_end);
            //     // info!("has annotated intron: {}", has_annotated_intron);
            //     info!("win1_A: {:?}\t{:?},{}/{}", win1_a_fraction, std::str::from_utf8(&major_bases_win1).unwrap(), win1_matched_A_count, window_size);
            //     info!("win1_T: {:?}\t{:?},{}/{}", win1_t_fraction, std::str::from_utf8(&major_bases_win1).unwrap(), win1_matched_T_count, window_size);
            //     info!("win2_A: {:?}\t{:?},{}/{}", win2_a_fraction, std::str::from_utf8(&major_bases_win2).unwrap(), win2_matched_A_count, window_size);
            //     info!("win2_T: {:?}\t{:?},{}/{}", win2_t_fraction, std::str::from_utf8(&major_bases_win2).unwrap(), win2_matched_T_count, window_size);
            //     info!("igv region: {}:{}-{},{}:{}-{}", chr, win1_start + 1, win1_end, chr, win2_start + 1, win2_end);
            //     info!("ref pos (1-based, inclusive): {}\t{}", start + 1, end);
            //     info!("ref seq: {:?}\t{:?}", std::str::from_utf8(&win1_ref_seq), std::str::from_utf8(&win2_ref_seq));
            //     info!("read seq: {:?}\t{:?}", std::str::from_utf8(&win1_read_seq), std::str::from_utf8(&win2_read_seq));
            //     info!("{}:{}, {}:{}", begin_cigar.unwrap().char(), begin_cigar.unwrap().len(), end_cigar.unwrap().char(), end_cigar.unwrap().len());
            //     // info!("Annotated intron: {}", annotated_intron);
            //     if (is_polyT_win1 && win1_t_fraction >= fraction) && win1_dist > end_distance {
            //         info!("discarded by win1");
            //     } else if (is_polyA_win2 && win2_a_fraction >= fraction) && win2_dist > end_distance {
            //         info!("discarded by win2");
            //     } else {
            //         info!("not discarded");
            //     }
            // }

            // Discard the records: polyA signal + far from known transcript ends (2 conditions must be satisfied)
            if (is_polyT_win1 && win1_t_fraction >= fraction) && win1_dist > end_distance {
                discarded_records.push(record);
            } else if (is_polyA_win2 && win2_a_fraction >= fraction) && win2_dist > end_distance {
                discarded_records.push(record);
            } else {
                out_records.push(record);
            }

            if (is_polyT_win1 && win1_t_fraction >= fraction) && win1_dist > end_distance {
                head_feature.label = 1;
            }
            if (is_polyA_win2 && win2_a_fraction >= fraction) && win2_dist > end_distance {
                tail_feature.label = 1;
            }

            // // Discard the records: polyA signal + no annotated intron + far from known transcript ends (3 conditions must be satisfied)
            // if (is_polyT_win1 && win1_t_fraction >= fraction) && (!has_annotated_intron && win1_dist > end_distance) {
            //     discarded_records.push(record);
            // } else if (is_polyA_win2 && win2_a_fraction >= fraction) && (!has_annotated_intron && win2_dist > end_distance) {
            //     discarded_records.push(record);
            // } else {
            //     out_records.push(record);
            // }
        }
        out_features.push(head_feature);
        out_features.push(tail_feature);
    }
    (out_records, discarded_records, out_features)
}