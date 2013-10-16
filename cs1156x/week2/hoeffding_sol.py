#!/usr/bin/env python

import sys
import math
import random
import os
import numpy

if len(sys.argv) < 2:
    print "3\nNumber of Tests\t100000\nNumber of Coins\t1000\nFlip\t10"
    sys.exit(0)

#This function is a stub for your code for the coin flipping simulation, here it just returns three random values for v_1, v_rand, and v_min
def flip_coins (coins, flip):
    flips = []
    for c in range(coins):
        res = numpy.random.random_integers (0, 1, flip)
        flips.append(numpy.average(res))
    vone = flips[0]
    vrnd = flips[random.randint (0, coins - 1)]
    vmin = min(flips)
    return (vone, vrnd, vmin)

parameters = [float(x) for x in sys.argv[1:-2]]
row_id = int(sys.argv[-2])
out_file = sys.argv[-1]
tmp_file = out_file + ".tmp"

tests = int (parameters[0])
coins = int (parameters[1])
flip = int (parameters[2])

fout = open (tmp_file, 'w')
fout.write ("Test::number,V_one::number,V_rnd::number,V_min::number\n")
for t in range (tests):
    vone, vrnd, vmin = flip_coins (coins, flip)
    fout.write (str(t) + ',' + str(vone) + ',' + str(vrnd) + ', '+ str(vmin) + '\n')
    if (t +1) % 1000 == 0:
        sys.stderr.write (str(t + 1) + " / " + str(tests) + "\n")
fout.close ()

os.rename (tmp_file, out_file)
    
    
