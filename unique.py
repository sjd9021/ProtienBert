def read_and_filter_kmers(input_filepath):
    """
    Reads k-mers from a file, filters out masked tokens, and returns a set of unique k-mers.
    """
    unique_kmers = set()
    with open(input_filepath, 'r') as file:
        for line in file:
            kmer = line.strip()
            if kmer != '[MASK]':  # Exclude masked tokens
                unique_kmers.add(kmer)
    return unique_kmers

def write_kmers_to_file(kmers, output_filepath):
    """
    Writes each k-mer in a set to a new line in the specified file.
    """
    with open(output_filepath, 'w') as file:
        for kmer in sorted(kmers):  # Write sorted k-mers for consistency
            file.write(kmer + '\n')

# Define file paths
input_filepath = 'kmers_masked3.txt'
output_filepath = 'kmers.txt'

# Process the k-mers
unique_kmers = read_and_filter_kmers(input_filepath)

# Write unique k-mers to file
write_kmers_to_file(unique_kmers, output_filepath)
print(f"Unique k-mers have been written to {output_filepath}.")
