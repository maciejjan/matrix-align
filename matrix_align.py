import argparse
from collections import Counter
import csv
import itertools
import logging
from operator import itemgetter
import torch
import tqdm
import sys
import time


### BEGIN SHORTSIM CODE
# this code is copied from:
# https://github.com/hsci-r/shortsim/blob/719605d1722fc75227dd66b8b9dacaa51eb2fe55/src/shortsim/ngrcos.py
# to avoid depending on the shortsim package.
# It is also converted to use torch instead of numpy.

def ngrams(string, n):
    return (string[i:i+n] for i in range(len(string)-n+1))


def determine_top_ngrams(strings, n, dim):
    ngram_freq = Counter()
    for text in strings:
        ngram_freq.update(ngrams(text, n))

    ngram_ids = {
        ngr : i \
        for i, (ngr, freq) in enumerate(sorted(
            ngram_freq.items(), key=itemgetter(1), reverse=True)[:dim]) }
    return ngram_ids


def vectorize(verses, n=2, dim=200, min_ngrams=10, ngram_ids=None,
              normalize=True, weighting='plain'):

    if ngram_ids is None:
        ngram_ids = determine_top_ngrams(map(itemgetter(1), verses), n, dim)

    v_ids, v_texts, rows = [], [], []
    for (v_id, text) in verses:
        v_ngr_ids = [ngram_ids[ngr] for ngr in ngrams(text, n) \
                     if ngr in ngram_ids]
        if len(v_ngr_ids) >= min_ngrams:
            row = torch.zeros(dim, dtype=torch.float32)
            for ngr_id in v_ngr_ids:
                row[ngr_id] += 1
            rows.append(row)
            v_ids.append(v_id)
            v_texts.append(text)
    m = torch.vstack(rows)
    if weighting == 'sqrt':
        m = torch.sqrt(m)
    elif weighting == 'binary':
        m = torch.asarray(m > 0, dtype=torch.float32)
    if normalize:
        m = torch.divide(m.T, torch.linalg.norm(m, axis=1)).T
    return v_ids, v_texts, ngram_ids, m

### END SHORTSIM CODE

def read_input(filename):
    verses = []
    with open(filename) as fp:
        reader = csv.DictReader(fp)
        for line in reader:
            verses.append(((line['poem_id'], line['pos']), line['text']))
    return verses


def similarity(x, y, yb, threshold=0.5, sim_raw_thr=0,
               rescale=False, return_alignments=False):
    d = torch.mm(y, x.T)
    if rescale:
        d = (d-threshold) / threshold
        d[d < 0] = 0
    else:
        d[d < threshold] = 0
    # similarity -> alignment matrix
    d[yb[:-1]] = torch.cummax(d[yb[:-1]], 1).values
    idx = yb[:-1]+1
    idx[torch.isin(idx, yb, assume_unique=True)] = 0
    while torch.any(idx > 0):
        idx = idx[idx > 0]
        d[idx,0] = torch.fmax(d[idx-1,0], d[idx,0])
        d[idx,1:d.shape[1]] = torch.fmax(
            d[idx-1,1:d.shape[1]],
            d[idx-1, 0:(d.shape[1]-1)]+d[idx,1:d.shape[1]])
        d[idx,] = torch.cummax(d[idx,], 1).values
        idx += 1
        idx[torch.isin(idx, yb, assume_unique=True)] = 0
    if not return_alignments:
        return d[yb[1:]-1,-1]
    else:
        # extract the aligned pairs
        # `a` contains for each row in `y` the index of the aligned
        #     counterpart in `x`, or -1 if not aligned
        # `w` contains the weights of the aligned pairs
        a = -torch.ones(d.shape[0], dtype=torch.long)
        w = torch.zeros(d.shape[0], dtype=d.dtype)
        # The alignments are extracted for all poems simultaneously.
        # (i, j) will contain the indices of the currently processed cells
        # In each step, we will advance each index to the next cell
        # on the best-alignment path.
        # If the end of the alignment is reached, the indices are removed.
        i = yb[1:]-1
        i = i[d[i,-1] > sim_raw_thr]
        j = torch.ones(i.shape[0], dtype=torch.long) * (d.shape[1]-1)
        if x.device.type == 'cuda':
            # if computing on the GPU -- move also the newly created
            # vectors to the GPU
            a, w, j = a.cuda(), w.cuda(), j.cuda()
        while i.shape[0] > 0:
            # NOT uppermost row of a poem
            u = torch.isin(i, yb[:-1], assume_unique=True, invert=True).int()
            v = (j > 0).int()                     # NOT leftmost column
            q = (d[i,j] == d[i-1,j] * u)          # did we come from above?
            r = (d[i,j] == d[i,j-1] * v) * (~q)   # did we come from the left?
            s = (~q) * (~r)                       # did we come from up-left?
            a[i[s]] = j[s]
            w[i[s]] = d[i[s],j[s]] - d[i[s]-1, j[s]-1] * u[s] * v[s]
            i = i - q.int() - s.int()
            j = j - r.int() - s.int()
            # keep the indices if:
            # - we have not crossed a poem boundary (the current i is not the
            #   last row of a poem OR we came from the right)
            # - both i and j are > 0
            keep = (torch.isin(i, yb[1:]-1, assume_unique=True, invert=True) + r) \
                   * (i >= 0) * (j >= 0)
            i = i[keep]
            j = j[keep]
        return d[yb[1:]-1,-1], a, w


def compute_similarities(
        m, poem_boundaries,
        threshold=0.5, sim_raw_thr=2.0,
        sim_onesided_thr=0.1, sim_sym_thr=0,
        rescale=False, return_alignments=False, print_progress=False):

    n = len(poem_boundaries)-1
    pbar = tqdm.tqdm(total=n, ncols=70) if print_progress else None
    
    for i in range(n):
        sim_result = similarity(m[poem_boundaries[i]:poem_boundaries[i+1],],
                                m[poem_boundaries[i+1]:],
                                poem_boundaries[(i+1):]-poem_boundaries[i+1],
                                threshold=threshold, rescale=rescale,
                                return_alignments=return_alignments,
                                sim_raw_thr=sim_raw_thr)
        (sim_raw, a, w) = sim_result if return_alignments \
                          else (sim_result, None, None)
        p1_length = poem_boundaries[i+1]-poem_boundaries[i]
        sim_l = sim_raw / p1_length
        p2_lengths = poem_boundaries[(i+2):]-poem_boundaries[(i+1):-1]
        sim_r = sim_raw / p2_lengths
        sim_sym = 2*sim_raw / (p2_lengths + p1_length)
        for j in torch.argwhere((sim_raw > sim_raw_thr) \
                                & ((sim_l > sim_onesided_thr) \
                                    | (sim_r > sim_onesided_thr)) \
                                & (sim_sym > sim_sym_thr)
                               ).flatten():
            als = None
            if return_alignments:
                a_j = a[poem_boundaries[i+j+1]-poem_boundaries[i+1]:\
                        poem_boundaries[i+j+2]-poem_boundaries[i+1]]
                w_j = w[poem_boundaries[i+j+1]-poem_boundaries[i+1]:\
                        poem_boundaries[i+j+2]-poem_boundaries[i+1]]
                als = [(int(a_j[k]), int(k), float(w_j[k])) \
                       for k in torch.where(a_j > -1)[0]]
            yield (i, int(i+j+1), float(sim_raw[j]), float(sim_l[j]),
                   float(sim_r[j]), float(sim_sym[j]), als)
        # update the progress bar
        if pbar is not None:
            pbar.update()


def format_als_for_output(als, p1_idx, p2_idx, poem_ids, v_ids, v_texts,
                          add_texts=False):
    p1_id = poem_ids[p1_idx]
    p2_id = poem_ids[p2_idx]
    if add_texts:
        return itertools.chain(
                ((p1_id, v_ids[poem_boundaries[p1_idx]+pos1][1],
                  v_texts[poem_boundaries[p1_idx]+pos1],
                  p2_id, v_ids[poem_boundaries[p2_idx]+pos2][1],
                  v_texts[poem_boundaries[p2_idx]+pos2], w) \
                 for pos1, pos2, w in als),
                ((p2_id, v_ids[poem_boundaries[p2_idx]+pos2][1],
                  v_texts[poem_boundaries[p2_idx]+pos2],
                  p1_id, v_ids[poem_boundaries[p1_idx]+pos1][1], \
                  v_texts[poem_boundaries[p1_idx]+pos1], w)
                 for pos1, pos2, w in als))
    else:
        return itertools.chain(
                ((p1_id, v_ids[poem_boundaries[p1_idx]+pos1][1],
                  p2_id, v_ids[poem_boundaries[p2_idx]+pos2][1], w) \
                 for pos1, pos2, w in als),
                ((p2_id, v_ids[poem_boundaries[p2_idx]+pos2][1],
                  p1_id, v_ids[poem_boundaries[p1_idx]+pos1][1], w) \
                 for pos1, pos2, w in als))


def setup_logging(logfile, level):
    if logfile is None:
        logging.basicConfig(level=level,
                            format='%(asctime)s %(message)s',
                            datefmt='%d.%m.%Y %H:%M:%S')
    else:
        logging.basicConfig(filename=logfile, level=level,
                            format='%(asctime)s %(message)s',
                            datefmt='%d.%m.%Y %H:%M:%S')


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Align large collections of versified poetry.')
    parser.add_argument(
        '-a', '--alignments-file', type=str, default=None,
        help='File to write verse-level alignments to.')
    parser.add_argument(
        '-d', '--dim', type=int, default=450,
        help='The number of dimensions of n-gram vectors for verses')
    parser.add_argument(
        '-g', '--use-gpu', action='store_true',
        help='Use the GPU for computation.')
    parser.add_argument(
        '-i', '--input-file', type=str, default=None,
        help='Input file (CSV: poem_id, pos, text)')
    parser.add_argument('--logfile', metavar='FILE')
    parser.add_argument('-L', '--logging-level', metavar='LEVEL',
                        default='WARNING', 
                        choices=['ERROR', 'WARNING', 'INFO', 'DEBUG'])
    parser.add_argument(
        '-m', '--min-ngrams', type=int, default=10,
        help='Minimum number of known n-grams to consider a verse.')
    parser.add_argument(
        '-n', type=int, default=2,
        help='The size (`n`) of the n-grams (default: 2, i.e. ngrams).')
    parser.add_argument(
        '-o', '--output-file', type=str, default=None,
        help='Output file.')
    parser.add_argument(
        '-p', '--print-progress', action='store_true',
        help='Print a progress bar.')
    parser.add_argument(
        '-r', '--rescale', action='store_true',
        help='After applying the threshold, rescale the verse similarities'
             ' to [0, 1].')
    parser.add_argument(
        '-t', '--threshold', type=float, default=0.5,
        help='Minimum verse cosine similarity to consider (default=0.5).')
    parser.add_argument(
        '-T', '--print-texts', action='store_true',
        help='Print texts of the aligned verses.')
    parser.add_argument(
        '-w', '--weighting', choices=['plain', 'sqrt', 'binary'],
        default='plain', help='Weighting of n-gram frequencies.')
    parser.add_argument(
        '--sim-raw-thr', type=float, default=2.0,
        help='Threshold on raw similarity (default=2).')
    parser.add_argument(
        '--sim-onesided-thr', type=float, default=0.1,
        help='Threshold on one-sided similarity (default=0.1).')
    parser.add_argument(
        '--sim-sym-thr', type=float, default=0,
        help='Threshold on symmetric similarity (default=0).')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    setup_logging(args.logfile, args.logging_level)
    
    verses = read_input(args.input_file)
    
    logging.info('starting vectorization')
    v_ids, v_texts, ngram_ids, m = \
        vectorize(verses, n=args.n, min_ngrams=args.min_ngrams, dim=args.dim,
                  weighting=args.weighting)
    logging.info('vectorization completed')
    poem_boundaries = [0] \
        + [i+1 for i in range(len(v_ids)-1) if v_ids[i][0] != v_ids[i+1][0]] \
        + [len(v_ids)]
    poem_ids = [v_ids[i][0] for i in poem_boundaries[:-1]]
    
    if args.use_gpu:
        logging.info('using torch on GPU')
        import torch.cuda
        poem_boundaries_a = torch.tensor(poem_boundaries).cuda()
        m = torch.tensor(m, dtype='float16').cuda()
    else:
        logging.info('using torch on CPU')
        poem_boundaries_a = torch.tensor(poem_boundaries)
    
    logging.info('starting similarity computation')
    t1 = time.time()

    sims = compute_similarities(
        m, poem_boundaries_a,
        threshold=args.threshold,
        rescale=args.rescale,
        print_progress=args.print_progress,
        return_alignments=(args.alignments_file is not None),
        sim_raw_thr=args.sim_raw_thr,
        sim_onesided_thr=args.sim_onesided_thr,
        sim_sym_thr=args.sim_sym_thr,
    )
    
    alfp, a_writer = None, None
    if args.alignments_file is not None:
        alfp = open(args.alignments_file, 'w+')
        a_writer = csv.writer(alfp, delimiter=',', lineterminator='\n')
        if args.print_texts:
            a_writer.writerow(('p1_id', 'pos1', 'text1',
                               'p2_id', 'pos2', 'text2', 'sim'))
        else:
            a_writer.writerow(('p1_id', 'pos1', 'p2_id', 'pos2', 'sim'))

    if args.output_file is None:
        writer = csv.writer(sys.stdout, delimiter=',', lineterminator='\n')
        writer.writerow(('p1_id', 'p2_id', 'sim_raw', 'sim_l', 'sim_r', 'sim'))
        for p1_idx, p2_idx, sim_raw, sim_l, sim_r, sim, als in sims:
            writer.writerow((poem_ids[p1_idx], poem_ids[p2_idx],
                             sim_raw, sim_l, sim_r, sim))
            writer.writerow((poem_ids[p2_idx], poem_ids[p1_idx],
                             sim_raw, sim_r, sim_l, sim))
            if a_writer is not None:
                rows = format_als_for_output(
                  als, p1_idx, p2_idx,
                  poem_ids, v_ids, v_texts, add_texts=args.print_texts)
                a_writer.writerows(rows)
    else:
        with open(args.output_file, 'w+') as outfp:
            writer = csv.writer(outfp, delimiter=',', lineterminator='\n')
            writer.writerow(('p1_id', 'p2_id', 'sim_raw', 'sim_l', 'sim_r', 'sim'))
            for p1_idx, p2_idx, sim_raw, sim_l, sim_r, sim, als in sims:
                writer.writerow((poem_ids[p1_idx], poem_ids[p2_idx],
                                 sim_raw, sim_l, sim_r, sim))
                writer.writerow((poem_ids[p2_idx], poem_ids[p1_idx],
                                 sim_raw, sim_r, sim_l, sim))
                if a_writer is not None:
                    rows = format_als_for_output(
                      als, p1_idx, p2_idx,
                      poem_ids, v_ids, v_texts, add_texts=args.print_texts)
                    a_writer.writerows(rows)

    if alfp:
        alfp.close()

    t2 = time.time()
    logging.info('similarity computation completed in {} s'.format(t2-t1))

