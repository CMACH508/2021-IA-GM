from os import system

for i in range(1,102):
  q = (3-len(str(i)))*'0' + str(i)
  system("python delaunay.py hotels/hotel" + str(i) + " > hotel" + q + ".adj")

