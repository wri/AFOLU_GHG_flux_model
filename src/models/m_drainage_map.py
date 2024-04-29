"""
This script executes the first step of the model (after preprocessing) which involves creating a binary map of drainage
 (1 = drained, 0 = undrained)
"""


import argparse

def drainage_map():
    #parser logic is here for example. replace with logic relevant to model
    parser = argparse.ArgumentParser(description='Read a file and print its content.')
    parser.add_argument('--file', '-f', required=True, help='Path to the file to read.')
    args = parser.parse_args()

    with open(args.file, 'r') as f:
        content = f.read()
        print(content)

if __name__ == '__main__':
    drainage_map()
