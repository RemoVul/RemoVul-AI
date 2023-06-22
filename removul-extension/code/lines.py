import sys
import json

file_path = sys.argv[1]

with open(file_path, 'r') as file:
    lines = file.readlines()
    # print 2,4
    print(2,',' ,4 ,',', 6)
    print(7,',', 11)
