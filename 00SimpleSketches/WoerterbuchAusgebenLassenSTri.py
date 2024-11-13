targets = [
    {'Anton': [1, 2, 3]},
    {1: [4, 5, 6]},
    {2: [7, 8, 9]}  # Dieses Element hat den Index 2
]

print("abc", targets[0])     #{0: [1, 2, 3]}
print("abc", targets[1])
print("abc", targets[2])
anton_values = targets[0]['Anton']
print(anton_values)

for target in targets:
    boxes = target["Anton"]

print("boxes", boxes)