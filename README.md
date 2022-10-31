# matrix-align

This repository contains code for the optimized computation of
line-by-line alignment of versified poetry. The method was described in
the following paper:

Maciej Janicki. Optimizing the weighted sequence alignment algorithm
for large-scale text similarity computation. In: *Proceedings of the
2nd International Workshop on Natural Language Processing for Digital
Humanities – NLP4DH 2022*.

Currently the repository contains only the optimized PyTorch
implementation (for both CPU and GPU), without the less optimal variants
used in the benchmarks.

## Installation and running

To install the dependencies, run:
```
pip3 install -r requirements.txt
```

Then you can run the `matrix_align.py` script. In order to see a
description of available parameters, run:
```
python3 matrix_align.py --help
```

## Example

The input to the script should be a CSV file with one verse per line
and three columns: `poem_id`, `pos` (position of the verse in the poem),
`text`, e.g.:
```
poem_id,pos,text
kalevala01,2,mieleni_minun_tekevi
kalevala01,3,aivoni_ajattelevi
kalevala01,4,lähteäni_laulamahan
kalevala01,5,saaani_sanelemahan
```

The file `example_input.csv` contains a small demo corpus comprised
of the publicly available works
[Kalevala](https://www.gutenberg.org/ebooks/7000),
[Old Kalevala](https://www.gutenberg.org/ebooks/48380)
and [Kanteletar](https://www.gutenberg.org/ebooks/7078).
The works were split into chapters/poems and verses and cleaned
(lowercased, removed punctuation and numbers).

Run the following command:
```
python3 matrix_align.py \
  -i example_input.csv -o example_output.csv -a example_alignments.csv \
  -L INFO -p -T
```

The computation should take around 5 minutes and will create two output
files. `example_output.csv` contains pairs of similar poems, with the
following columns:
* `p1_id`, `p2_id` -- poem IDs
* `sim_raw` -- raw similarity, i.e. the sum of cosine similarities of aligned verses,
* `sim_l`, `sim_r` -- left-/right-normalized similarity (the raw similarity divided by the length of the left/right poem)
* `sim` -- symmetric normalized similarity (harmonic mean of the left- and right-normalized)

The file `example_alignments.csv` contains pairs of aligned lines in
the following format:
* `p1_id`, `pos1` -- poem ID and position of the first line,
* `text1` -- text of the first line (if running with `-T` parameter)
* `p2_id`, `pos2` -- poem ID and position of the second line,
* `text2` -- text of the second line (if running with `-T` parameter)
* `sim` -- n-gram cosine similarity.

Note that the alignments file is only produced if the `-a` parameter is
set, and that without it the computation is faster as the alignments do
not need to be extracted.

### Scalability

In practice, the script can be used on much larger collections, such as
[Suomen Kansan Vanhat Runot](https://skvr.fi/) ([Github](https://github.com/sks190/SKVR))
or [Eesti Regilaulude Andmebaas](https://www.folklore.ee/regilaul/andmebaas/).

## BERT example

The core of the script is the function `similarity(x, y, yb)` which conducts
the optimized alignment on embedding matrices `x` and `y`, where `yb`
is a vector of document boundaries in the matrix `y`. This algorithm
can be applied regardless of which vectorization method was used to obtain
the embeddings (provided that the vectors are normalized to unit length,
so that cosine similarity amounts to a dot product).

The file `bert_demo.py` contains a toy example illustrating how to use
the alignment method on vectors obtained from BERT (using the HuggingFace
`transformers` library). The alignment captures similar phrases, e.g.
`quick brown fox ~ swift red squirrel`.

A more realistic example illustrating the benefit of large-scale alignment
based on BERT embeddings is subject for further work.
