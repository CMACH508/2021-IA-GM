from PIL import Image, ImageDraw
import random
import math
import numpy
import sys
from os import system

ind = sys.argv[1]

imname = "alex." + ind + ".png"
cname = "alex." + ind + ".corner"
aname = "alex." + ind + ".adj"
cornname = "alex." + ind + ".list"

im = Image.open(imname)
corns = [tuple([float(c) for c in b.split()]) for b in open(cname, 'r').readlines()]
adj = [tuple([int(c) for c in b.split()]) for b in open(aname, 'r').readlines()]
correct = [int(c) for c in open(cornname, 'r').read().strip().split()]

draw = ImageDraw.Draw(im)

for i in range(len(correct)-1):
  draw.line(list(corns[correct[i]]) + list(corns[correct[i+1]]), fill=255)

im.save("junk.png")

for i in range(len(corns)):
    for j in range(i,len(corns)):
      if (adj[i][j] == 1):
        draw.line(list(corns[i]) + list(corns[j]), fill=128)

im.save("junk2.png")


