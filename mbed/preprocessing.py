from mbed.trimming import edge_lengths, find_edges
from mbed.generation import make_line_mesh
import networkx as nx
from math import ceil
import numpy as np


def stitch(mesh, edges):
    '''Remove edge and stitch together vertices'''
    assert mesh.topology().dim() == 1

    vertices = list(range(mesh.num_vertices()))
    # We want to rewrite this such that given edges are 
    # kept but based on their connectivity we perform stitches
    # changing other cells
    e2v = mesh.topology()(1, 0)
    # NOTE: by saying that edge is to be removed we're saying that its
    # vertices can be regarded as same. A piece of graph of connected
    # edges is thus a substitution rule that all vertices in that piece
    # can be condensed
    g = nx.Graph()
    g.add_edges_from(map(e2v, edges))

    rules = {}
    # Substitution rule
    for cc in sorted(nx.algorithms.connected_components(g), key=len):
        substitute = cc.pop()
        for node in cc:
            assert node not in rules
            rules[node] = substitute

    # Edges are definitely part of the new mesh. Some other cells might
    # be added as well if substitution produces two equal nodes
    edges = list(edges)
    # We want to keep a mapping from new mesh cell_idx to what was approx
    # the parent cells in the oritinal mesh
    cell_map, new_cells = [], []
    # Now we rewrite
    cells = mesh.cells().tolist()
    for cell_id, cell in enumerate(cells):
        if cell_id not in edges:
            # Subs
            if cell[0] in rules: cell[0] = rules[cell[0]]
            if cell[1] in rules: cell[1] = rules[cell[1]]
            # Invalid
            cell[0] == cell[1] and edges.append(cell_id)
            # A valid cell will keep it's position in new mesh
            if cell[0] != cell[1]:
                cell_map.append(cell_id)
                new_cells.append(cell)

    # Which vertices will be used; this is also a map from new to old
    vertex_map = list(set(sum(new_cells, [])))
    # Need to finally rewrte the cells this way; so vertex old -> new needed
    ivertex_map = {o: n for n, o in enumerate(vertex_map)}
    new_cells = np.fromiter((ivertex_map[v] for v in sum(new_cells, [])), dtype='uintp').reshape((-1, 2))

    new_coordinates = mesh.coordinates()[vertex_map]

    return make_line_mesh(new_coordinates, new_cells), cell_map, vertex_map


    # edges, rewritten_edges, removed_vertices = set(edges), set(), set()
    # for edge in edges:
    #     if edge not in rewritten_edges:
    #         v_remove, v_keep = sorted(e2v(edge), key=lambda e: len(e2v(e)))
    #         removed_vertices.add(v_remove)
            
    #         rewrite_edges = set(v2e(v_remove))

    #         for e in rewrite_edges:
    #             if e in rewritten_edges: continue

    #             cell = cells[e]
    #             if v_remove in cell:
    #                 print cell, (v_remove, v_keep), 
    #                 cell[e2v(e).tolist().index(v_remove)] = v_keep
    #                 print '->', cell
    #                 rewritten_edges.add(e)
    #     print
    # print cells



def refine(mesh, threshold):
    '''Refine such that the new mesh has cell size of at most dx'''
    assert mesh.topology().dim() == 1

    e2v = mesh.topology()(1, 0)

    cells = {old: [c.tolist()] for old, c in enumerate(mesh.cells())}
    x = mesh.coordinates()
    
    lengths = edge_lengths(mesh)
    needs_refine = find_edges(lengths, predicate=lambda v, x: v > threshold)

    next_v = len(x)
    for cell in needs_refine:
        v0, v1 = cells[cell].pop()
        x0, x1 = x[v0], x[v1]
        l = np.linalg.norm(x0 - x1)

        nodes = [v0]
        
        ts = np.linspace(0., 1., int(ceil(l/threshold))+1)[1:-1]
        dx = x1 - x0
        for t in ts:
            xmid = x0 + t*dx
            
            x = np.row_stack([x, xmid])
            
            nodes.append(next_v)
            next_v += 1

        nodes.append(v1)

        cells[cell] = list(zip(nodes[:-1], nodes[1:]))

    

    # Mapping for 
    parent_cell_map = sum(([k]*len(cells[k]) for k in sorted(cells)), [])

    cells = np.array(sum((cells[k] for k in sorted(cells)), []), dtype='uintp')
    mesh = make_line_mesh(x, cells)

    return mesh, np.array(parent_cell_map, dtype='uintp')
    
# --------------------------------------------------------------------

if __name__ == '__main__':
    coords = np.array([[0, 0], [1, 0], [1, 0.2], [1, 0.5], [1, 0.7], [1, 1], [0, 1.]])
    mesh = make_line_mesh(coords, close_path=True)

    rmesh, mapping = refine(mesh, threshold=0.6)

    x = mesh.coordinates()
    y = rmesh.coordinates()
    assert np.linalg.norm(x - y[:len(x)]) < 1E-13

    e2v, E2V = mesh.topology()(1, 0), rmesh.topology()(1, 0)
    for c in range(mesh.num_cells()):
        x0, x1 = x[e2v(c)]
        e = x1 - x0
        e = e/np.linalg.norm(e)

        for C in np.where(mapping == c)[0]:
            y0, y1 = y[E2V(C)]
            E = y1 - y0
            E = E/np.linalg.norm(E)

            assert abs(1-abs(np.dot(e, E))) < 1E-13

                                                      

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from mbed.generation import make_line_mesh

    vertices = np.array([[0., 0.],
                         [1., 0.],
                         [2, 0.],
                         [0., 1.],
                         [0., 2.],
                         [-3, -3]])
    
    cells = np.array([[0, 1], [0, 3], [1, 3], [3, 4], [1, 2], [0, 5]])

    mesh = make_line_mesh(vertices, cells)

    smesh = stitch(mesh, [0, 1, 2])[0]
    print smesh.num_cells()
