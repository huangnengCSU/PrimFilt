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
          Discarded output BAM file, optional
  -p, --primers-trimmed
          Primer sequences (e.g. oligo(dT)) are trimmed from reads, default: false
  -w, --window-size <WINDOW_SIZE>
          Window size for check interal priming [default: 20]
  -f, --fraction <FRACTION>
          Fraction of A's in the window to consider as internal priming, default: 0.7. Recommended: 0.7 for primer-trimmed reads, 0.6 for non-trimmed reads [default: 0.7]
  -e, --end-distance <END_DISTANCE>
          Maximum distance to known annotated transcript end [default: 100]
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
- **Primers trimmed**. If primer sequences are trimmed from reads, use the `--primers-trimmed` argument to indicate this.
```shell
PrimFilt -i input.bam -r reference.fa -o filtered.bam -d discarded.bam -w 20 -f 0.7 -t 8 -p
```

## Update Log
### 0.1.2 - 2026-01-23
- Added `--end-distance` argument to set the maximum allowed distance to a known transcript end.
- Implemented additional filtering logic for reads with internal priming signals: reads are discarded if they do not overlap any annotated intron, or if their distance to the nearest annotated transcript end exceeds the threshold (default: 100 bp). 

### 0.1.1 - 2026-01-20
- Added `--primers-trimmed` argument to indicate if primer sequences are trimmed from reads.
- Implemented logic for untrimmed reads: detect a poly(A)/poly(T) tail at the read end (≥10 nt with up to 1 error) and evaluate internal-priming signals accordingly.

### 0.1.0 - 2026-01-10
- Core internal priming detection functionality
- Optional annotation-aware filtering
- Multithreaded BAM processing


## Acknowledgement
PrimFilt is inspired by the talon_label_reads module in [TALON](https://github.com/mortazavilab/TALON). Building on this foundation, PrimFilt incorporates optional gene annotation support, enabling selective filtering of internally primed reads that are not associated with annotated intronic regions.

## License
MIT License

Copyright (c) 2026 Dana-Farber Cancer Institute

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.



