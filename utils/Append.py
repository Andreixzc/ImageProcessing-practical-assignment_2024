import csv

def append_csv(csv_file_x, csv_file_y, output_file):
    with open(csv_file_x, 'r') as infile_x, open(csv_file_y, 'r') as infile_y, open(output_file, 'w', newline='') as outfile:
        reader_x = csv.reader(infile_x)
        reader_y = csv.reader(infile_y)
        writer = csv.writer(outfile)

        # Read the header row of both CSV files
        header_x = next(reader_x)
        header_y = next(reader_y)

        # Append headers from both files
        writer.writerow(header_y + header_x)

        # Append rows from both files
        for row_x, row_y in zip(reader_x, reader_y):
            writer.writerow(row_y + row_x)

# Replace 'file_x.csv' with the path to your first CSV file
file_y = 'hu_treated.csv'
# Replace 'file_y.csv' with the path to your second CSV file
file_x = 'labelPath.csv'
# Replace 'output.csv' with the path to the output CSV file
output_csv = 'output.csv'

append_csv(file_x, file_y, output_csv)
