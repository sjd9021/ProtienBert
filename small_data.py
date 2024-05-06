def extract_and_save_subset(input_filepath, output_filepath, num_lines=90000):
    """ Extracts the first `num_lines` lines from `input_filepath` and saves them to `output_filepath`. """
    with open(input_filepath, 'r') as file:
        lines = [next(file).strip() for _ in range(num_lines)]
    
    with open(output_filepath, 'w') as file:
        for line in lines:
            file.write(line + '\n')

# Define your file paths
input_filepath = '/Users/admin/ProtienBert/kmers_masked3.txt'  # Update this to the path of your input file
output_filepath = '/Users/admin/ProtienBert/train_subset.txt'  # Update this to where you want to save the subset

# Call the function to extract and save the first 10000 lines
extract_and_save_subset(input_filepath, output_filepath, num_lines=90000)

print(f"The first 10000 lines have been extracted from {input_filepath} and saved to {output_filepath}.")
