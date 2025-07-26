"""
This script provides an efficient O(N log N) pure python implementation for 
computing the Kemeny distance between two order semantics.

The implementation is based on the decomposition of the distance into
discordant and partial discordant pairs, as described in the paper
'Order-Aware GSGP and its Application in Cross-Sectional Trading' by
Bunjerdtaweeporn & Moraglio.

The calculation is broken down into three components based on two functions:

    1. 'compute_discordant_pairs': Counts semantics pairs with different established 
        order. Implemented in O(N log N).

    2. 'compute_par': Counts semantic pairs that are tied in one but establish 
        order in the other. Implemented in O(N).

The main public functions are 'compute_kemeny_distance' and
                              'compute_normalised_kemeny_distance'.
"""

from collections import defaultdict
from typing import Sequence, Dict, List, Tuple

# Type alias for clarity
PredictionSemantics = Sequence[float]

def compute_discordant(semantics_a: PredictionSemantics, 
                       semantics_b: PredictionSemantics) -> int:
    """
    Calculate the number of discordant semantic pairs between 'semantics_a' and
    'semantics_b'. This implements the 'dis(s(g1),s(g2))' function in the paper. 
    The time complexity for this function is O(N log N).

    Parameters:
        semantics_a : The first prediction semantics vector.
        semantics_b : The second prediction semantics vector.

    Returns:
        The total number of discordant semantic pairs.
    """
    
    assert len(semantics_a) == len(semantics_b), 'Both vectors length should be identical'
    
    # Group indices by their semantic value in semantics_b.
    
    unique_groups_b: Dict[float, List[int]] = defaultdict(list)
    
    for i, semantic_val in enumerate(semantics_b):
        unique_groups_b[semantic_val].append(i)
    
    # Create a list of these groups, sorted by their semantic value.
    
    sorted_unique_groups_b = sorted(unique_groups_b.keys())
    
    # Within each group (tied semantic values in semantics_b), sort the indices
    # based on their corresponding values in semantics_a. This handles intra-group
    # comparisons correctly
    
    index_groups_b = [
        sorted(unique_groups_b[semantic], key=lambda i: semantics_a[i])
        for semantic in sorted_unique_groups_b
        ]
    
    def merge_and_count(left_group: List[int], 
                        right_group: List[int]) -> Tuple[List[int], int]:
        """Merges two sorted groups of indices and counts inversions."""
        
        merged = []
        inversions = 0
        i = 0
        j = 0
        
        while i < len(left_group) and j < len(right_group):
            
            if semantics_a[left_group[i]] <= semantics_a[right_group[j]]:
                merged.append(left_group[i])
                i += 1
            else: # An inversion semantic pair is found
                # All remaining elements in left_group are also > right_group[j]
                inversions += len(left_group) - i
                merged.append(right_group[j])
                j += 1
        
        merged.extend(left_group[i:])
        merged.extend(right_group[j:])
        
        return merged, inversions
    
    def recursive_merge_count(groups: List[List[int]]) -> Tuple[List[int], int]:
        """Recursively splits, sorts, and counts inversions between groups."""
        
        if not groups:
            return [], 0
        
        if len(groups) == 1:
            return groups[0], 0

        mid = len(groups) // 2
        left, left_inversions = recursive_merge_count(groups[:mid])
        right, right_inversions = recursive_merge_count(groups[mid:])

        merged, merge_inversions = merge_and_count(left, right)
        
        return merged, left_inversions + right_inversions + merge_inversions
    
    _, total_discordant = recursive_merge_count(index_groups_b)
    
    return total_discordant

def compute_par(semantics_a: PredictionSemantics, 
                semantics_b: PredictionSemantics) -> int:
    """
    Calculate the number of semantic pairs that are tied in 'semantics_b' but 
    establish order in 'semantics_a'. This implements the 'par(s(g1),s(g2))' 
    function in the paper. The time complexity for this function is O(N).

    Parameters:
        semantics_a : The first prediction semantics vector.
        semantics_b : The second prediction semantics vector.

    Returns:
        The number of semantic pairs that are tied in 'semantics_b' but 
        establish order in 'semantics_a'.
    """
    
    # Group indices by their semantic value in semantics_b.
    
    unique_groups_b: Dict[float, List[int]] = defaultdict(list)
    
    for i, semantic_val in enumerate(semantics_b):
        unique_groups_b[semantic_val].append(i)
        
    partial_discordant_count = 0
    
    # Iterate through each semantic group from semantics_b
    for indices in unique_groups_b.values():
        
        # Total semantic pair that are tied in semantics_b for this semantic group
        num_indices = len(indices)
        total_tied_pairs_b = num_indices * (num_indices - 1) // 2
        
        # Calculate how many of these semantic pairs remain tied in semantics_a
        subgroups_a: Dict[float, int] = defaultdict(int)
        
        for index in indices:
            subgroups_a[semantics_a[index]] += 1
        
        # Sum of semantic pairs that are also tied in semantics_a
        tied_pairs_in_a = sum(
            count * (count - 1) // 2 for count in subgroups_a.values()
        )
        
        # The difference gives the semantic pairs that are tied in 'b' but ordered in 'a'
        partial_discordant_count += total_tied_pairs_b - tied_pairs_in_a
    
    return partial_discordant_count

def compute_kemeny_distance(semantics_a: PredictionSemantics, 
                            semantics_b: PredictionSemantics) -> int:
    """
    Calculate the Kemeny distance with an efficient O(N log N) algorithm.
    
    Parameters:
        semantics_a : The first prediction semantics vector.
        semantics_b : The second prediction semantics vector.

    Returns:
        The Kemeny distance value.
    """

    dis = compute_discordant(semantics_a, semantics_b)
    par_1 = compute_par(semantics_a, semantics_b)
    par_2 = compute_par(semantics_b, semantics_a)

    return 2 * dis + par_1 + par_2

def compute_normalised_kemeny_distance(semantics_a: PredictionSemantics, 
                                       semantics_b: PredictionSemantics) -> float:
    """
    Calculate the normalised Kemeny distance with an efficient O(N log N) algorithm.
    
    Parameters:
        semantics_a : The first prediction semantics vector.
        semantics_b : The second prediction semantics vector.

    Returns:
        The normalised Kemeny distance, scaled to the range [0, 1].
    """
    kemeny_distance = compute_kemeny_distance(semantics_a, semantics_b)
    n = len(semantics_a)
    
    normalised_kemeny_distance = kemeny_distance / (n*(n-1))
    
    return normalised_kemeny_distance

def naive_kemeny_distance(semantics_a: PredictionSemantics, 
                          semantics_b: PredictionSemantics) -> int:
    """
    A straightforward O(N^2) implementation of Kemeny distance for verification

    Parameters:
        semantics_a : The first prediction semantics vector.
        semantics_b : The second prediction semantics vector.

    Returns:
        The Kemeny distance value.

    """
    assert len(semantics_a) == len(semantics_b), 'Input vectors must have the same length.'
    
    kemeny_distance = 0
    n = len(semantics_a)
    
    def sign(a, b):
        if a > b:
            return 1
        elif a < b:
            return -1
        elif a == b:
            return 0
        
    for i in range(n):
        for j in range(n):
            ord_ij_semantics_a = sign(semantics_a[i], semantics_a[j])
            ord_ij_semantics_b = sign(semantics_b[i], semantics_b[j])
            kemeny_distance += 0.5*abs(ord_ij_semantics_a - ord_ij_semantics_b) 
    
    return kemeny_distance
    
def naive_normalised_kemeny_distance(semantics_a: PredictionSemantics, 
                                     semantics_b: PredictionSemantics) -> float:
    """
    A straightforward O(N^2) implementation of normalised Kemeny distance for verification
    
    Parameters:
        semantics_a : The first prediction semantics vector.
        semantics_b : The second prediction semantics vector.

    Returns:
        The normalised Kemeny distance, scaled to the range [0, 1].

    """
    
    kemeny_distance = naive_kemeny_distance(semantics_a, semantics_b)
    n = len(semantics_a)
    
    normalised_kemeny_distance = kemeny_distance / (n*(n-1))
    
    return normalised_kemeny_distance
    