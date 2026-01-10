# PrimFilt
A long-read RNA-seq internal priming detection and filtering tool.

## Installation
```shell
git clone https://github.com/huangnengCSU/PrimFilt.git
cd PrimFilt
cargo build --release
```
The executable will be located at `target/release/PrimFilt`.

## Usage
```
Usage: PrimFilt [OPTIONS] --input <INPUT> --reference <REFERENCE> --output <OUTPUT>

Options:
  -i, --input <INPUT>
          Input BAM file
  -r, --reference <REFERENCE>
          Input REF file
  -a, --annotation <ANNOTATION>
          Input Annotation file, optional
  -o, --output <OUTPUT>
          Filtered output BAM file
  -d, --discarded-output <DISCARDED_OUTPUT>
          Discarded output BAM file
  -w, --window-size <WINDOW_SIZE>
          Window size for check interal priming [default: 20]
  -f, --fraction <FRACTION>
          Fraction of A's in the window to consider as internal priming [default: 0.7]
  -t, --threads <THREADS>
          Number of threads to use [default: 1]
  -h, --help
          Print help
```
- **Not providing annotation**. PrimFilt takes two windows at the front and back of the aligned read and check for the fraction of A's and T's respectively. If the fraction exceeds the default threshold 70%, the read is considered as internally primed.
```shell
PrimFilt -i input.bam -r reference.fa -o filtered.bam -d discarded.bam -w 20 -f 0.7 -t 8
```
- **Providing annotation**. If annotation is provided, PrimFilt will filter out only those reads that have no any annotated intron and show internal priming signals.
```shell
PrimFilt -i input.bam -r reference.fa -a annotation.gtf -o filtered.bam -d discarded.bam -w 20 -f 0.7 -t 8
```

## Acknowledgement
PrimFilt is inspired by the talon_label_reads module in [TALON](https://github.com/mortazavilab/TALON). Building on this foundation, PrimFilt extends the original approach by incorporating optional gene annotation support, enabling selective filtering of internally primed reads that are not associated with annotated intronic regions. 



