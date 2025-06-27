from typing import List, Tuple
from Bio import pairwise2
from Bio.SubsMat.MatrixInfo import blosum62

# Default gap penalties
GAP_OPEN = -10
GAP_EXTEND = -0.5
#Blosum-based similarity distance
def blosum_raw_score(seq1: str, seq2: str,
                     matrix: dict = blosum62,
                     gap_open: float = GAP_OPEN,
                     gap_extend: float = GAP_EXTEND) -> float:
    """
    Compute the best global alignment score between seq1 and seq2
    using the given substitution matrix and affine gap penalties.
    """
    alignments = pairwise2.align.globalds(seq1, seq2, matrix,
                                          gap_open, gap_extend,
                                          one_alignment_only=True)
    return alignments[0].score

def normalized_blosum_similarity(seq1: str, seq2: str,
                                 matrix: dict = blosum62,
                                 gap_open: float = GAP_OPEN,
                                 gap_extend: float = GAP_EXTEND
                                ) -> float:
    """
    Returns the normalized BLOSUM‐based similarity in [0,1]:
    
        Sim_pct(seq1,seq2) = S(seq1,seq2) / max{ S(seq1,seq1), S(seq2,seq2) }
    """
    # Raw cross‐score
    s_ab = blosum_raw_score(seq1, seq2, matrix, gap_open, gap_extend)
    # Self‐scores for normalization
    s_aa = blosum_raw_score(seq1, seq1, matrix, gap_open, gap_extend)
    s_bb = blosum_raw_score(seq2, seq2, matrix, gap_open, gap_extend)
    
    denom = max(s_aa, s_bb)
    if denom <= 0:
        # Avoid division by zero; fallback to zero similarity
        return 0.0
    return s_ab / denom

def blosum_distance(seq1: str, seq2: str, **kwargs) -> float:
    """
    Converts normalized similarity into a distance in [0,1]:
    
        d = 1 - Sim_pct
    """
    sim = normalized_blosum_similarity(seq1, seq2, **kwargs)
    return 1.0 - sim

def build_distance_matrix(seqs: List[str], **kwargs) -> List[List[float]]:
    """
    Given a list of sequences, returns the full NxN distance matrix,
    where entry [i][j] = blosum_distance(seqs[i], seqs[j]).
    """
    n = len(seqs)
    D = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            d = blosum_distance(seqs[i], seqs[j], **kwargs)
            D[i][j] = D[j][i] = d
    return D

def seq_identity(seq1: str, seq2: str) -> float:
    """
    Compute the fraction of identical positions in the best global alignment
    of seq1 and seq2, using simple match=1, mismatch=0.
    Returns a value in [0,1].
    """
    # globalxx: match score=1, mismatch=0, and default gap penalties
    align = pairwise2.align.globalxx(seq1, seq2, one_alignment_only=True)[0]
    aln1, aln2 = align.seqA, align.seqB
    matches = sum(a == b for a, b in zip(aln1, aln2))
    length = len(aln1)  # includes gaps
    return matches / length if length > 0 else 0.0
#Sequence Identity based Matrix
def seq_distance(seq1: str, seq2: str) -> float:
    """
    Distance based on sequence identity:
        d = 1 - SeqID(seq1, seq2)
    """
    return 1.0 - seq_identity(seq1, seq2)

def build_seqid_distance_matrix(seqs: List[str]) -> List[List[float]]:
    """
    Given a list of sequences, returns the full NxN distance matrix,
    where entry [i][j] = seq_distance(seqs[i], seqs[j]).
    """
    n = len(seqs)
    D = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            d = seq_distance(seqs[i], seqs[j])
            D[i][j] = D[j][i] = d
    return D

