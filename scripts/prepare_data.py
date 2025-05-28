input_file = r"C:\Users\Neksi\Desktop\F1-winner-model\f1_ready_for_model.csv"
output_file = r"C:\Users\Neksi\Desktop\F1-winner-model\outputs\f1_ready_for_model_cleaned.csv"

# Count commas in header for row validation
with open(input_file, "r", encoding="utf-8", errors="ignore") as f_in:
    header = f_in.readline()
    expected_commas = header.count(",")

# Filter and clean
with open(input_file, "r", encoding="utf-8", errors="ignore") as f_in, \
     open(output_file, "w", encoding="utf-8") as f_out:
    
    f_out.write(header)  # keep header
    for line in f_in:
        if line.count(",") == expected_commas:
            f_out.write(line)

print("✅ Cleaned file saved as f1_ready_for_model_cleaned.csv")



