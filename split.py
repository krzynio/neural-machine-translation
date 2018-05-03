#!/usr/bin/env python3

import sys

with open(sys.argv[1]) as src:
    with open(sys.argv[2], 'w') as first:
        with open(sys.argv[3], 'w') as second:
            first_size = int(sys.argv[4])
            for line in src:
                if first_size > 0:
                    first.write(line)
                    first_size -=1
                else:
                    second.write(line)
