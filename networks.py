from collections import defaultdict
from collections import Counter
from abc import ABC, abstractmethod


class Network(ABC):
    """An abstract class that enforces implementation of methods common for all networks"""
    @abstractmethod
    def __init__(self, directed):
        self.directed = directed

    @abstractmethod
    def update_edge(self, from_node, to_node):
        pass

    @abstractmethod
    def get_num_edges(self):
        pass


class DirectedNetwork(Network):
    """Implements a data structure used to represent the user
     interactions (retweets, quotes, replies) - directed weighted graph"""
    def __init__(self):
        super().__init__(True)
        self.num_connections = 0
        self.adj_dict = defaultdict(Counter)

    def update_edge(self, from_node, to_node):
        if from_node == to_node:
            return
        self.num_connections += 1
        self.adj_dict[from_node][to_node] += 1

    def get_num_edges(self):
        num_edges = 0
        for edges in self.adj_dict.values():
            num_edges += len(edges)
        return num_edges

    def get_triads(self):
        triads = []
        closed_triads = []
        for node in self.adj_dict.keys():
            for conn in self.adj_dict[node].keys():
                if conn not in self.adj_dict:
                    continue
                for conn2 in self.adj_dict[conn].keys():
                    if node != conn2:
                        triad = (node, conn, conn2)
                        triads.append(triad)
                        if conn2 in self.adj_dict.keys() and node in self.adj_dict[conn2].keys():
                            closed_triads.append(triad)
        return triads, closed_triads


class UndirectedNetwork(Network):
    """Implements a data structure used to represent the co-occurrence of hashtags - undirected unweighted graph"""
    def __init__(self):
        super().__init__(False)
        self.adj_dict = defaultdict(set)
        self.all_nodes = set()
        self.num_edges = 0
        self.clique_count = 0

    def update_edge(self, from_node, to_node):
        if from_node == to_node:
            return

        self.all_nodes.update({from_node, to_node})

        if to_node not in self.adj_dict[from_node]:
            self.num_edges += 1
        self.adj_dict[from_node].add(to_node)
        self.adj_dict[to_node].add(from_node)

    def connect_all_vertices(self, vertices):
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                self.update_edge(vertices[i], vertices[j])

    def get_num_edges(self):
        return self.num_edges

    def find_cliques(self):
        if len(self.adj_dict) == 0:
            return
        # vertices = set(self.adj_dict.keys())
        self.clique_count = 0
        self.bron_kerbosh(list(), list(self.all_nodes), list(), depth=0, limit=20)

    def bron_kerbosh(self, r, p, x, depth, limit):
        if depth > 7:
            return

        if (len(p) == 0 and len(x) == 0) and depth > 4:
            print(r)
            self.clique_count += 1
            return

        for node in p:
            neighs = list(self.adj_dict[node]) if node in self.adj_dict else list()
            new_r = r + [node]
            new_p = [v for v in p if v in neighs]
            new_x = [v for v in x if v in neighs]
            self.bron_kerbosh(new_r, new_p, new_x, depth+1, limit)
            if self.clique_count == limit:
                return

            p.remove(node)
            x.append(node)
