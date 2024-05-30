import csv

def remove_substring_from_csv(input_file, output_file, substring):
    with open(input_file, 'r', newline='') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        for row in reader:
            # Remove the substring from each element in the row
            new_row = [element.replace(substring, '') for element in row]
            # Write the modified row to the new file
            writer.writerow(new_row)

# Example usage
input_file = 'labelPath.csv'
output_file = 'output.csv'
substring = '/home/joaop/PUC/Curso_CC_6p/PAI/Trabalhos/out/'

remove_substring_from_csv(input_file, output_file, substring)
