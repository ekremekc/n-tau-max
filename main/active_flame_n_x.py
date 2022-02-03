import dolfinx
import basix
from dolfinx.fem  import Function, FunctionSpace, Constant
from dolfinx.geometry import compute_collisions, compute_colliding_cells, BoundingBoxTree
from mpi4py import MPI
from ufl import Measure,  TestFunction, TrialFunction, inner
from petsc4py import PETSc
import numpy as np

class ActiveFlame:

    gamma = 1.4

    def __init__(self, mesh, subdomains, x_r, rho_u, Q, U, n , tau, omega, degree=1):

        self.mesh = mesh
        self.subdomains = subdomains
        self.x_r = x_r
        self.rho_u = rho_u
        self.Q = Q
        self.U = U
        self.n = n
        self.tau = tau
        self.omega = omega
        self.degree = degree

        self.coeff = (self.gamma - 1) / rho_u * Q / U

        # __________________________________________________

        self._a = {}
        self._b = {}
        self._D_kj = None
        self._D_kj_adj = None
        self._D = None
        self._D_adj = None

        # __________________________________________________

        self.V = FunctionSpace(mesh, ("Lagrange", degree))


        self.u = TrialFunction(self.V)
        self.v = TestFunction(self.V)

        for fl, x in enumerate(self.x_r):
            # print(fl,x)
            self._a[str(fl)] = self._assemble_left_vector(fl)
            self._b[str(fl)] = self._assemble_right_vector(x)
        

    @property
    def submatrices(self):
        return self._D_kj

    @property
    def matrix(self):
        return self._D
    @property
    def a(self):
        return self._a
    @property
    def b(self):
        return self._b        
    @property
    def adjoint_submatrices(self):
        return self._D_kj_adj

    @property
    def adjoint_matrix(self):
        return self._D_adj

    def _assemble_left_vector(self, fl):
        """
        Assembles \int v(x) \phi_k dV
        Parameters
        ----------
        fl : int
            flame tag
        Returns
        -------
        v : <class 'tuple'>
            includes assembled elements of a
        """

        dx = Measure("dx", subdomain_data=self.subdomains)

        phi_k = self.v
        

        V_fl = MPI.COMM_WORLD.allreduce(dolfinx.fem.assemble_scalar(Constant(self.mesh, PETSc.ScalarType(1))*dx(fl)), op=MPI.SUM)
        b = Function(self.V)
        b.x.array[:] = 0
        const = Constant(self.mesh, (1/V_fl))

        n = self.n
        tau = self.tau 
        tau_func = Function(FunctionSpace(self.mesh, ("Lagrange", self.degree)))
        #print(len(tau_func.x.array[:]), len(np.exp(self.omega*1j*tau.x.array)))
        tau_func.x.array[:] = np.exp(self.omega*1j*tau.x.array) 

        a = dolfinx.fem.assemble_vector(b.vector, n * tau_func * inner(const, phi_k) * dx(fl))
        b.vector.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        indices1 = np.array(np.flatnonzero(a.getArray()),dtype=np.int32)
        a = b.x.array
        dofmaps = self.V.dofmap
        global_indices = dofmaps.index_map.local_to_global(indices1)
        a = list(zip(global_indices, a[indices1]))
        # print("A", a)
        return a

    def _assemble_right_vector(self, point):
        """
        Calculates degree of freedoms and indices of 
        right vector of
        
        \nabla(\phi_j(x_r)) . n
        
        which includes gradient value of test fuunction at
        reference point x_r
        
        Parameters
        ----------
        x : np.array
            flame location vector
        Returns
        -------
        np.array
            Array of degree of freedoms and indices as vector b.
        """
        tdim = self.mesh.topology.dim

        v = np.array([[0, 0, 1]]).T
        if tdim == 1:
            v = np.array([[1]])
        elif tdim == 2:
            v = np.array([[1, 0]]).T

        # Finds the basis function's derivative at point x
        # and returns the relevant dof and derivative as a list
        bb_tree = BoundingBoxTree(self.mesh, tdim)
        cell_candidates = compute_collisions(bb_tree, point)

        # Choose one of the cells that contains the point
        if tdim == 1:
            if len(cell_candidates.array)>0: # For 1D Parallel Runs
                cell = [cell_candidates.array[0]]
            else:
                cell = []
        else:
            cell = compute_colliding_cells(self.mesh, cell_candidates, point)
        
        # Data required for pull back of coordinate
        gdim = self.mesh.geometry.dim
        num_local_cells = self.mesh.topology.index_map(tdim).size_local
        num_dofs_x = self.mesh.geometry.dofmap.links(0).size  # NOTE: Assumes same cell geometry in whole mesh
        t_imap = self.mesh.topology.index_map(tdim)
        num_cells = t_imap.size_local + t_imap.num_ghosts
        
        x = self.mesh.geometry.x
        x_dofs = self.mesh.geometry.dofmap.array.reshape(num_cells, num_dofs_x)
        cell_geometry = np.zeros((num_dofs_x, gdim), dtype=np.float64)
        points_ref = np.zeros((1, tdim)) #INTERESTINGLY REQUIRED FOR PARALLEL RUNS

        # Data required for evaluation of derivative
        ct = self.mesh.topology.cell_type
        element = basix.create_element(basix.finite_element.string_to_family( #INTERESTINGLY REQUIRED FOR PARALLEL RUNS
        "Lagrange", ct.name), basix.cell.string_to_type(ct.name), self.degree, basix.LagrangeVariant.equispaced)
        dofmaps = self.V.dofmap
        coordinate_element = basix.create_element(basix.finite_element.string_to_family(
                "Lagrange", ct.name), basix.cell.string_to_type(ct.name), 1, basix.LagrangeVariant.equispaced)

        point_ref = None
        B = []
        if len(cell) > 0:
            # Only add contribution if cell is owned
            
            cell = cell[0]
            
            if cell < num_local_cells:
                # Map point in cell back to reference element
                cell_geometry[:] = x[x_dofs[cell], :gdim]
                point_ref = self.mesh.geometry.cmap.pull_back([point[:gdim]], cell_geometry)
                dphi = coordinate_element.tabulate(1, point_ref)[1:,0,:]
                dphi = dphi.reshape((dphi.shape[0], dphi.shape[1]))
                J = np.dot(cell_geometry.T, dphi.T)
                Jinv = np.linalg.inv(J)  

                cell_dofs = dofmaps.cell_dofs(cell)
                global_dofs = dofmaps.index_map.local_to_global(cell_dofs)
                # Compute gradient on physical element by multiplying by J^(-T)
                d_dx = (Jinv.T @ dphi).T
                d_dv = np.dot(d_dx, v)[:, 0]
                for i in range(len(d_dv)):
                    B.append([global_dofs[i], d_dv[i]])
            else:
                print(MPI.COMM_WORLD.rank, "Ghost", cell) 
        root = 0 #it was -1
        if len(B) > 0:
            root = MPI.COMM_WORLD.rank
        b_root = MPI.COMM_WORLD.allreduce(root, op=MPI.MAX)
        B = MPI.COMM_WORLD.bcast(B, root=b_root)
        # print("B ",B)
        return B

    @staticmethod
    def _csr_matrix(a, b):

        # len(a) and len(b) are not the same

        nnz = len(a) * len(b)
        

        row = np.zeros(nnz)
        col = np.zeros(nnz)
        val = np.zeros(nnz, dtype=np.complex128)

        for i, c in enumerate(a):
            for j, d in enumerate(b):
                row[i * len(b) + j] = c[0]
                col[i * len(b) + j] = d[0]
                val[i * len(b) + j] = c[1] * d[1]

        row = row.astype(dtype='int32')
        col = col.astype(dtype='int32')
        # print("ROW: ",row,
        # "COL: ",col,
        # "VAL: ",val)
        return row, col, val

    def assemble_submatrices(self, problem_type='direct'):
        """
        This function handles efficient cross product of the 
        vectors a and b calculated above and generates highly sparse 
        matrix D_kj which represents active flame matrix without FTF and
        other constant multiplications.
        Parameters
        ----------
        problem_type : str, optional
            Specified problem type. The default is 'direct'.
            Matrix can be obtained by selecting problem type, other
            option is adjoint.
        
        """

        num_fl = len(self.x_r)  # number of flames
        global_size = self.V.dofmap.index_map.size_global
        local_size = self.V.dofmap.index_map.size_local
 
        # print("LOCAL SIZE: ",local_size, "GLOBAL SIZE: ", global_size)

        row = dict()
        col = dict()
        val = dict()

        for fl in range(num_fl):

            u = None
            v = None

            if problem_type == 'direct':
                u = self._a[str(fl)]
                v = self._b[str(fl)]

            elif problem_type == 'adjoint':
                u = self._b[str(fl)]
                v = self._a[str(fl)]

            row[str(fl)], col[str(fl)], val[str(fl)] = self._csr_matrix(u, v)

        row = np.concatenate([row[str(fl)] for fl in range(num_fl)])
        col = np.concatenate([col[str(fl)] for fl in range(num_fl)])
        val = np.concatenate([val[str(fl)] for fl in range(num_fl)])

        i = np.argsort(row)

        row = row[i]
        col = col[i]
        val = val[i]
        
        # print("ROW: ",repr(row))
        # print("COLUMN: ",repr(col))
        # print("VAL: ",repr(val))
        if len(val)==0:
            
            mat = PETSc.Mat().create(comm=PETSc.COMM_WORLD) #PETSc.COMM_WORLD
            mat.setSizes([(local_size, global_size), (local_size, global_size)])
            mat.setFromOptions()
            mat.setUp()
            mat.assemble()
            
        else:
            # indptr = np.bincount(row, minlength=local_size)
            # indptr = np.insert(indptr, 0, 0).cumsum()
            # indptr = indptr.astype(dtype='int32')
            mat = PETSc.Mat().create(PETSc.COMM_WORLD) # MPI.COMM_SELF
            mat.setSizes([(local_size, global_size), (local_size, global_size)])
            mat.setType('aij') 
            mat.setUp()
            # mat.setValuesCSR(indptr, col, val)
            for i in range(len(row)):
                mat.setValue(row[i],col[i],val[i], addv=PETSc.InsertMode.ADD_VALUES)
            mat.assemblyBegin()
            mat.assemblyEnd()

        # print(mat.getValues(range(global_size),range(global_size)))
        if problem_type == 'direct':
            self._D = mat*self.coeff
        elif problem_type == 'adjoint':
            self._D = mat*self.coeff
        # print("Matrix D generated.")

    
