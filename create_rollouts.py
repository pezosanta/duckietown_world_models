import os
import sys

if len(sys.argv) < 4:
    print("usage: %s train test validation" %sys.argv[0])
    sys.exit(0)

cwd = os.getcwd()

# Creating required directory structure
if os.path.isdir("duckietown_utils/datasets") == False:
    os.mkdir(cwd + "/duckietown_utils/datasets")
if os.path.isdir("duckietown_utils/datasets/duckie") == False:
    os.mkdir(cwd + "/duckietown_utils/datasets/duckie")
if os.path.isdir("duckietown_utils/datasets/images") == False:
    os.mkdir(cwd + "/duckietown_utils/datasets/images")

# Training dataset generation
os.system("python3.6 generator.py %d 1 0" %int(sys.argv[1]))

# Test dataset generation
os.system("python3.6 generator.py %d 0 0" %int(sys.argv[2]))

# Validation dataset generation
os.system("python3.6 generator.py %d 2 0" %int(sys.argv[3]))
