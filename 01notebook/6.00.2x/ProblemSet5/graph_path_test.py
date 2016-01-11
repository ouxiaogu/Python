# https://www.python.org/doc/essays/graphs/
graph = {'A': ['B', 'C', 'H'],
     'B': ['C', 'D'],
     'C': ['D', 'F', 'H'],
     'D': ['C'],
     'E': ['F', 'H'],
     'F': ['C'],
     'H': ['D']}


def find_all_paths(graph, start, end, path=[]):
    path = path + [start]
    print path
    if start == end:
        return [path]
    if not graph.has_key(start):
        # print start+' []'
        return []
    paths = []
    for node in graph[start]:
        if node not in path:
            newpaths = find_all_paths(graph, node, end, path)
            for newpath in newpaths:
                paths.append(newpath)
    return paths

print find_all_paths(graph, 'A', 'D')
