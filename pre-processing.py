import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_sequences_only(filepath):
    sequences = []
    valid_chars = set("ACDEFGHIKLMNPQRSTVWYX*-")  # Valid amino acid symbols and common placeholders
    try:
        with open(filepath, 'r') as file:
            for line in file:
                seq = line.strip().upper()  # Remove newline and convert to upper case
                if all(char in valid_chars for char in seq):
                    sequences.append(seq)
                else:
                    logging.warning(f"Invalid characters found in sequence and will be skipped.")
    except IOError as e:
        logging.error(f"Error reading file {filepath}: {e}")
        return []
    
    return sequences

def write_sequences_to_txt(sequences, output_filepath):
    try:
        with open(output_filepath, 'w') as file:
            for sequence in sequences:
                file.write(sequence + '\n')  # Write each sequence on a new line
        logging.info(f"Sequences have been written to {output_filepath}")
    except IOError as e:
        logging.error(f"Error writing to file {output_filepath}: {e}")

def main(input_file, output_file):
    logging.info("Starting to parse sequences.")
    sequences = parse_sequences_only(input_file)
    logging.info(f"Successfully parsed {len(sequences)} sequences.")
    
    logging.info("Writing sequences to text file.")
    write_sequences_to_txt(sequences, output_file)

if __name__ == "__main__":
    input_file = 'parsed_sequences.txt'  # Path to your actual text file containing sequences
    output_file = 'parsed_sequences.txt'  # Output text file path
    main(input_file, output_file)
