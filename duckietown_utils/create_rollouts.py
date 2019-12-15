import os

cwd = os.getcwd()

# Creating required directory structure
os.mkdir(cwd + "/datasets")
os.mkdir(cwd + "/datasets/duckie")
os.mkdir(cwd + "/datasets/images")

# Training dataset generation
os.system("python3 generator.py 100 1 0")

# Test dataset generation
os.system("python3 generator.py 50 0 0")

# Validation dataset generation
os.system("python3 generator.py 20 2 0")
