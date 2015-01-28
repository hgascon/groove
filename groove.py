#!/usr/bin/python
# groove - a graph tool for vector embeddings
# Hugo Gascon <hgascon@mail.de>


import sys
import os
import argparse


def print_logo():
    print("""


   __ _ _ __ ___   _____   _____
  / _` | '__/ _ \ / _ \ \ / / _ \\
 | (_| | | | (_) | (_) \ V /  __/
  \__, |_|  \___/ \___/ \_/ \___|
   __/ |
  |___/                  v0.1-dev

            """)


def exit():
    print_logo()
    parser.print_help()
    sys.exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A Graph Tool for\
                                                  Vector Embeddings')

    parser.add_argument("-c", "--conf", default="groove/conf/example.conf",
                        help="Select configuration file.\
                        If no file is given, the default file\
                        'groove/conf/example.conf' will be read.")

    parser.add_argument("-d", "--dir", default="",
                        help="Load graph objects/files from this directory.")

    parser.add_argument("-o", "--out", default="",
                        help="Select output file.")

    args = parser.parse_args()
    path_conf = os.path.realpath(args.conf)

    exit()
