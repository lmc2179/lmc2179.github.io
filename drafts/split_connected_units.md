idea: find way of breaking up units in way that minimizes sutva violations

```python
import networkx as nx

def split_into_two_minimally_connected_sets(G):
    """Split a graph into two minimally connected sets."""
    cut = nx.minimum_cut(G, 0, len(G) - 1, capacity='weight')[1]
    return cut[0], cut[1]

def unit_test():
    """Unit test."""
    # Create a graph
    G = nx.Graph()
    # Add edges
    G.add_edge(0, 1, weight=1)
    G.add_edge(1, 2, weight=1)
    G.add_edge(2, 3, weight=1)
    G.add_edge(3, 4, weight=1)
    G.add_edge(4, 5, weight=10)
    G.add_edge(5, 0, weight=1)
    # Split the graph into two minimally connected sets
    left, right = split_into_two_minimally_connected_sets(G)
    # Print the sets
    print(left)
    print(right)

if __name__ == "__main__":
    unit_test()
```
