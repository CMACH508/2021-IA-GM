# Copyright (c) 2006, National ICT Australia
# All rights reserved.
#
# The contents of this file are subject to the Mozilla Public License Version
# 1.1 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
# http://www.mozilla.org/MPL/
#
# Software distributed under the License is distributed on an "AS IS" basis,
# WITHOUT WARRANTY OF ANY KIND, either express or implied. See the License
# for the specific language governing rights and limitations under the
# License.
#
# Author: Julian McAuley (j.mcauley@student.unsw.edu.au)
# Last Updated: 13/12/2006


## \package elefant.crf.beliefprop.core.structures.utilities
# A utility module.
# 
# Some basic utilities (many have been removed in favour of Python's  existing
# libraries).

from __future__ import division

import copy
import heapq
import sets



##################################################
### Kruskal's Algorithm                        ###
##################################################

## A forest.
#
# Stores a set of trees (a forest), and provides methods to  join  them.  This
# can be used as part of Kruskal's algorithm to find a spanning tree.
class Forest(object):
  ## Initialiser
  #
  # @param vertexlist A list of elements that will  be  the  vertices  of  our
  #   graph. Initially, each tree is just a single vertex.
  def __init__(self,
               vertexlist):
    # Maps each vertex to the tree that contains it.
    vertextree = {}
    # Maps each tree to a list of its vertices.
    treevertices = {}

    ntrees = 0

    for v in vertexlist:
      vertextree[v] = ntrees
      treevertices[ntrees] = [v]
      ntrees += 1

    self._vertextree = vertextree
    self._treevertices = treevertices
    self._ntrees = ntrees

  ## Get the number of disjoint trees.
  #
  # Get the number of trees in the graph that are disjoint. Once this  method
  # returns 1, no further progress will be possible, and we will have found a
  # spanning tree.
  def getntrees(self):
    return self._ntrees

  def gettrees(self):
    return self._treevertices.values()

  ## Try and join two vertices.
  #
  # @param v1, v2 A pair of vertices, both of which should be  in  our  vertex
  #   list
  # 
  # Join the two vertices if possible  (and  return  True),  otherwise  return
  # False (if joining them would create a loop).
  def join(self, v1, v2):
    if (self._vertextree[v1] == self._vertextree[v2]):
      return False
    
    t1 = self._vertextree[v1]
    t2 = self._vertextree[v2]

    for v in self._treevertices[t2]:
      self._vertextree[v] = t1
    self._treevertices[t1] += self._treevertices[t2]
    del self._treevertices[t2]

    self._ntrees -= 1

    return True



##################################################
### Cartesian Product                          ###
##################################################

## Calculate the cartesian product of a set of lists.
#
# @param domains A list of lists, each of which is the domain of one variable.
# 
# Calculate the cartesian product of multiple lists.
# 
# e.g. cp([1,2], [3,4,5]) will return [(1, 3), (2, 3), (1, 4), (2, 4), (1, 5),
# (2, 5)]
def cp(*domains):
  product = [()]
  for d in domains:
    newproduct = []
    for p in product:
      for e in d:
        newproduct.append( p + (e,) )
    product = newproduct
  return product


