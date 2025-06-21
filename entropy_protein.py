import numpy as np
from collections import Counter
from math import log2
import pandas as pd

def calculate_entropy(sequences):
    # Number of sequences (n) and length of each sequence (length)
    n = len(sequences)
    length = len(sequences[0])

    # List to store the entropy for each position
    entropy_list = []

    # Loop through each position 
    #ent_sum = 0  
    for i in range(length):
        # Extract the amino acids at position i from all sequences
        column = [seq[i] for seq in sequences]
        
        # Count the occurrences of each amino acid
        freq = Counter(column)
        
        # Calculate the probability of each amino acid
        prob = {aa: count / n for aa, count in freq.items()}
        
        # Calculate the entropy for this position
        entropy = -sum(p * log2(p) for p in prob.values() if p > 0)
        
        # Append the entropy of this position to the list
        entropy_list.append(entropy)

    
    return sum(entropy_list)

# Example

sequences = [
    "ACDEFGHKKLMNPQRSTVWXY",
    "ACFEFGHAKLMNPQRSTVWXS",
    "ACDEFGHIKLMNPQRSTVWXY"
    # Add more sequences as needed
]
#Extract the sequences from csv file
df = pd.read_csv('Bits_to_Binders_sequences_466max.csv')

# 2. Extract the “sequences” column into a Python list
sequences_list = df['sequence'].tolist()

#positive_sequences = [
    #[abs(x) for x in seq]
    #for seq in sequences_list
#]

# Calculate the entropy for each position in the sequences
entropy_values = calculate_entropy(sequences_list)
#print(len(entropy_values))
#print(sum(entropy_values))

grouped_sequences = df.groupby('file')['sequence'].apply(list).to_dict()

result = (
    df
    .groupby('file')['sequence']       # group by file, grab the sequences column
    .apply(list)                        # turn each group's values into a list
    .apply(calculate_entropy)                 # apply your numeric function
    .reset_index(name='total_characters')  # get a DataFrame with columns file & total_characters
)

print(result)



sequences_list 