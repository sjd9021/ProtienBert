from Bio import SeqIO

def parse_sequences_only(filepath):
    sequences = []
    for seq_record in SeqIO.parse(filepath, "fasta"):
        sequences.append(str(seq_record.seq))
    return sequences

def write_sequences_to_txt(sequences, output_filepath):
    # Open a file in write mode
    with open(output_filepath, 'w') as file:
        for sequence in sequences:
            file.write(sequence + '\n')  # Write each sequence on a new line

# Replace 'influenza.faa' with the path to your actual FASTA file
fasta_file = 'influenza.faa'
sequences = parse_sequences_only(fasta_file)

# Specify the path for the output text file
txt_file = 'parsed_sequences.txt'
write_sequences_to_txt(sequences, txt_file)

print(f"Sequences have been written to {txt_file}")
