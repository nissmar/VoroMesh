import numpy as np

fin = open("src/eval/result_thingi.txt", 'r')
lines = fin.readlines()
fin.close()
names = [e[:-2] for e in lines[::2]]
numbers = np.array([[float(i.strip()) for i in line.split()]
                    for line in lines[1::2]])


def print_single_line(ind):
    '''name, CD, F1, NC'''
    row = numbers[ind]
    print('{} {:.3f} {:.3f} {:.3f}'.format(
        names[ind], row[5]*100000, row[11], row[8]))  # CD


print("name, CD, F1, NC")
print_single_line(0)
print_single_line(1)
print_single_line(2)
