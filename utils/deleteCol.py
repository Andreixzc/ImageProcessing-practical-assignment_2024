import csv

def delete_last_column(csv_file, output_file):
    with open(csv_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        for row in reader:
            # Delete the last element of each row
            del row[-1]
            writer.writerow(row)

# Replace 'input.csv' with the path to your input CSV file
input_csv = 'hu_moments.csv'
# Replace 'output.csv' with the path to the output CSV file
output_csv = 'hu_treated.csv'

delete_last_column(input_csv, output_csv)
