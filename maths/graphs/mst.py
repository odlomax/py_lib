import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import lil_matrix
from scipy.spatial import Delaunay
from maths.graphs.edges import graph_edge_positions, graph_edge_lengths

class euclidean_mst:
    
    """
    
    Class to build euclidean minimum spanning tree
    
    Data
    ----
    
    del_tri: Delaunay triangulation object
        Qhull Delaunay triangulation object
        
    r[:,:]: float
        array of point positions [n_points:n_dim]
    
    
    tri: sparse matrix
        Delaunay triangulation sparse matrix
    
    mst: sparse matrix
        minimum spanning tree sparse matrix
    """
    
    def __init__(self,r):
        
        """
        Subroutine: build Euclidean minimum spanning tree from points
        
        Arguments
        ---------
        r[:,:]: float
            array of point positions [n_points:n_dim]
        
        """
        
        # Calculate Delaunay triangulation
        print("Generating Delaunay triangulation.")
        self.r=r
        self.del_tri=Delaunay(self.r)
        
        # Convert Delaunay triangulation to matrix graph format
        print("Converting triangulation to sparse matrix format.")
        self.tri=lil_matrix((r.shape[0],r.shape[0]),dtype=r.dtype)
        indices=self.del_tri.vertex_neighbor_vertices[0]
        indptr=self.del_tri.vertex_neighbor_vertices[1]
        
        # loop over all points
        for i in range(self.r.shape[0]):
            # loop over all neighbours of each point
            for j in indptr[indices[i]:indices[i+1]]:
                # only populate upper portion of del_tri
                if j>i:
                    # calculate edge weight of graph
                    l=np.linalg.norm(self.r[i,:]-self.r[j,:])
                    self.tri[i,j]=l
        
        # convert to csr format
        self.tri=self.tri.tocsr()
        
        # Calculate minimum spanning tree
        print("Generating minimum spanning tree.")
        self.mst=minimum_spanning_tree(self.tri)
        
    def tri_edge_positions(self):
        
        """
        Function: get edge line segments of Delaunay triangulation
        
        Result
        ------
        start[:,:],end[:,:]: float
            arrays of start and end points
        """
        
        return graph_edge_positions(self.tri,self.r)
        
    def mst_edge_positions(self):
        
        """
        Function: get edge line segments of minimum spanning tree
        
        Result
        ------
        start[:,:],end[:,:]: float
            arrays of start and end points
        """
        
        return graph_edge_positions(self.mst,self.r)
    
    def tri_edge_lengths(self):
        
        """
        Function: get edge lengths of Delaunay triangulation
        
        Result
        ------
        lengths[:]: float
            array of edge lengths
        """
        
        return graph_edge_lengths(self.tri)
    
    def mst_edge_lengths(self):
        
        """
        Function: get edge lengths of minimum spanning tree
        
        Result
        ------
        lengths[:]: float
            array of edge lengths
        """
        
        return graph_edge_lengths(self.mst)