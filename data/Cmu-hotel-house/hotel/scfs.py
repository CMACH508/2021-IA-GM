from os import system
import sys

for q in range(1,102):
  q2 = (3-len(str(q)))*'0' + str(q)
  system("./scfeatures none hotels/hotel" + str(q) + " hotel" + q2 + ".scf")

