# split_train_list.py
# Teilt eine .txt-Datei in "fake_train.txt" und "real_train.txt"
# basierend darauf, ob die Zeile mit "09" beginnt.

input_file = "train.txt"
fake_file = "fake_train.txt"
real_file = "real_train.txt"

with open(input_file, "r") as infile, \
     open(fake_file, "w") as f_fake, \
     open(real_file, "w") as f_real:
    
    for line in infile:
        line = line.strip()
        if not line:
            continue  # überspringe leere Zeilen
        
        if line.startswith("09"):
            f_fake.write(line + "\n")
        else:
            f_real.write(line + "\n")

print("✅ Aufgeteilt in:")
print(f"  - {fake_file}")
print(f"  - {real_file}")
