# This program, given an integer on the command line
# prints all the cycles in the complete graph of the given size
# Verified to have the right number for 4, 5, and 6. Got lazy
# after that.
#
# Imran Haque

import numpy
import sys

class Graph(object):
   def neighbors(self, v):
      return numpy.argwhere(self.adjacency[v,:] == 1)[:,0]

   def find_cycles_starting_at(self, v):
      """ General idea: any cycle involving v must involve one
      of its neighbors exactly once. Iterate over neighbors; each time, break
      the link from v to that neighbor, and find all paths from the neighbor
      back to v.
      
      In principle this has time complexity deg(v) * O(V+E) for the DFS steps
      alone but it's probably higher since I didn't account for the set
      operations or make any attempt at efficiency.
      """
      
      neighbors = self.neighbors(v)
      paths = set()
      for n in neighbors:
         # Delete the edge at the start
         self.adjacency[v,n] = 0
         self.adjacency[n,v] = 0
         # Initiate path-finding DFS from n to v
         #print "starting pathfinding from",v,"through",n
         nbr_paths = self.find_paths(n, v)
         if nbr_paths is None:
            continue
         for path in nbr_paths:
            edge_path = []
            src = v
            for nxt in path:
               # This is just a hack because I intended an undirected graph
               # but implemented using a directed adjacency matrix
               if src < nxt:
                  edge_path.append((src,nxt))
               else:
                  edge_path.append((nxt,src))
                  
               src = nxt
            edge_path = tuple(sorted(edge_path))
            paths.add(edge_path)
         # Restore the edge
         self.adjacency[v,n] = 1
         self.adjacency[n,v] = 1
      return paths

   def find_paths(self, s, t, path=[]):
      """ Find paths from s to t. 'path' is a recursion variable.
      """
      if s == t:
         path.append(t)
         return [path]
      neighbors = filter(lambda x: x not in path, self.neighbors(s))
      if len(neighbors) == 0:
         return None
      paths = []
      for v in neighbors:
         base = [x for x in path]
         base.append(s)
         new_paths = self.find_paths(v, t, base)
         if new_paths is not None:
            paths.extend(new_paths)
      return paths
   
class Kn(Graph):
   """ The fully connected graph on n vertices"""
   def __init__(self, n):
      self.vertices = range(n)
      self.adjacency = numpy.ones((n,n)) - numpy.eye(n)

class MoleculeGraph(Graph):
   """ The graph corresponding to a moleule"""
   def __init__(self, filename):
      """ Create graph corresponding to a specified molecule.
      """
      import openeye.oechem

      istream = openeye.oechem.oemolistream()
      istream.open(filename)
      molecule = openeye.oechem.OEGraphMol()
      openeye.oechem.OEReadMolecule(istream, molecule)
      istream.close()

      natoms = molecule.NumAtoms()
      self.vertices = range(natoms)
      self.adjacency = numpy.zeros([natoms, natoms], numpy.int8)
      for bond in molecule.GetBonds():
         iatom = bond.GetBgnIdx()
         jatom = bond.GetEndIdx()
         self.adjacency[iatom,jatom] = 1
         self.adjacency[jatom,iatom] = 1
      print self.adjacency
      print repr(self.adjacency)

def find_all_cycles(graph):
   cycles = set()
   for v in graph.vertices:      
      cycles = cycles | graph.find_cycles_starting_at(v)
   return cycles

def test():
   G1 = MoleculeGraph('molecules/benzene.mol2')
   C1 = find_all_cycles(G1)

   G2 = MoleculeGraph('molecules/naphthalene.mol2')      
   C2 = find_all_cycles(G2)

   # TODO: build grappphs for G1 and G2 with unified vertices using a specified mapping

   return
   
def main():
   #graph = Kn(int(sys.argv[1]))
   graph = MoleculeGraph(sys.argv[1])   
   cycles = set()
   for v in graph.vertices:      
      cycles = cycles | graph.find_cycles_starting_at(v)
   print cycles, len(cycles)
      
if __name__ == "__main__":
   main()
   #test()

