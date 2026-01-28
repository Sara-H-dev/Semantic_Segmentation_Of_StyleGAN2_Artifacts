with open("val.txt") as file:
    lines = [line.rstrip() for line in file]

real_counter = 0
fake_counter = 0

for line in lines:
    num = line[:2]
    if num != "09":
        real_counter += 1
    else:
        fake_counter +=1

print(f"Fake counter: {fake_counter}")
print(f"real counter: {real_counter}")
