#!/bin/bash

# Output Excel file
output_excel="all_model_stats.xlsx"

# Remove the output Excel file if it already exists
if [ -f "$output_excel" ]; then
    rm "$output_excel"
fi

# Flag to check if the Excel file has been created
excel_created=false

# Loop through all CSV files in the current directory
for csv_file in *.csv; do
    # Get the base name of the file (without extension)
    base_name=$(basename "$csv_file" .csv)
    
    # Run the Python script and capture the output
    echo "Processing $csv_file..."
    output=$(python3 compute_model_stats.py "$csv_file")
    
    # Check if the output is non-empty
    if [ -z "$output" ]; then
        echo "Warning: No data generated for $csv_file. Skipping..."
        continue
    fi
    
    # Save the output to a temporary CSV file
    echo "$output" > "${base_name}_stats.csv"
    
    # Determine the mode for ExcelWriter
    if [ "$excel_created" = false ]; then
        mode='w'
        excel_created=true
    else
        mode='a'
    fi
    
    # Convert the CSV output to an Excel sheet
    python3 -c "
import pandas as pd;
df = pd.read_csv('${base_name}_stats.csv');
with pd.ExcelWriter('$output_excel', mode='$mode', engine='openpyxl', if_sheet_exists='replace') as writer:
    df.to_excel(writer, sheet_name='$base_name', index=False);
"
    
    # Remove the temporary CSV file
    rm "${base_name}_stats.csv"
done

echo "All statistics saved to $output_excel"
