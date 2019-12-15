import os
import sys

if len(sys.argv) < 4:
    print("usage: %s train test validation" %sys.argv[0])
    sys.exit(0)

cwd = os.getcwd()

# Creating required directory structure
os.mkdir(cwd + "/datasets")
os.mkdir(cwd + "/datasets/duckie")
os.mkdir(cwd + "/datasets/images")

# Training dataset generation
os.system("python3 generator.py %d 1 0" %int(sys.argv[1]))

# Test dataset generation
os.system("python3 generator.py %d 0 0" %int(sys.argv[2]))

# Validation dataset generation
os.system("python3 generator.py %d 2 0" %int(sys.argv[3]))