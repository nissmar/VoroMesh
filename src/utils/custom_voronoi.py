"""Custom Voronoi diagram based on PyTorch"""
from torch import nn
import torch
import numpy as np
import igl
from scipy.spatial import Voronoi
from pytorch3d.ops import knn_points, knn_gather
import sys

try:
    sys.path.append("./src/cpp_utils/build/")
    from VoroMeshUtils import compute_voromesh, self_intersect
    CPP_COMPILED = True
except:
    print('WARNING: CGAL voromesh not found, using scipy mesh extraction with NO WATERTIGHTNESS GUARANTEES. Please compile cpp_utils.')
    CPP_COMPILED = False

SIGNS = np.array(
    [
        [(-1) ** i, (-1) ** j, (-1) ** k]
        for i in range(2)
        for j in range(2)
        for k in range(2)
    ]
)

# utilities


def face_orientation(p1, p2, p3, vp1, vp2):
    return (np.cross(p2 - p1, p3 - p1) * (vp1 - vp2)).sum() > 0


def mask_relevant_voxels(grid_n: int, samples: np.ndarray):
    """subselects voxels which collide with pointcloud"""
    samples_low = np.floor((samples + 1) / 2 * (grid_n - 1)).astype(np.int64)
    mask = np.zeros((grid_n, grid_n, grid_n))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                mask[
                    samples_low[:, 0] + i, samples_low[:, 1] +
                    j, samples_low[:, 2] + k
                ] += 1
    return mask.reshape((grid_n**3)) > 0


def voronoi_to_poly_mesh(vpoints: np.ndarray, interior_cells: np.ndarray, clip=True):
    """mesh voronoi diagram with marked interior cells"""
    pv = Voronoi(vpoints)
    faces = [0]

    prob_cnt = 0
    inds = []
    for face_, int1, int2 in zip(
        pv.ridge_vertices, pv.ridge_points[:, 0], pv.ridge_points[:, 1]
    ):
        face = face_.copy()
        if np.logical_xor(interior_cells[int1], interior_cells[int2]):
            if -1 in face:
                prob_cnt += 1
                print("WARNING: face ignored in the voronoi diagram")
            else:
                vp1 = pv.points[int1]
                vp2 = pv.points[int2]
                if interior_cells[int1]:
                    vp1, vp2 = vp2, vp1
                orient = face_orientation(
                    pv.vertices[face[0]],
                    pv.vertices[face[1]],
                    pv.vertices[face[2]],
                    vp1,
                    vp2,
                )
                faces.append(len(face) + faces[-1])

                if not (orient):
                    face.reverse()
                inds += face

    # select only the relevant vertices
    inds = np.array(inds)
    nvertices = pv.vertices.copy()
    un = np.unique(inds)
    inv = np.arange(inds.max() + 1)
    inv[un] = np.arange(len(un))
    nvertices = pv.vertices[un]
    inds = inv[inds]

    if clip:
        nvertices = np.maximum(-1, nvertices)
        nvertices = np.minimum(1, nvertices)

    is_erronious = bool(prob_cnt > 0)

    return (
        nvertices,
        [inds[faces[i]: faces[i + 1]].tolist() for i in range(len(faces) - 1)],
        is_erronious,
    )


def voronoi_to_mesh(
    vpoints: np.ndarray, interior_cells: np.ndarray, clip=True, return_errors=False
):
    """computes mesh from voronoi centers marked as inside or outside, with optional clip in [-1, 1]^3"""
    if CPP_COMPILED:
        vpoints = vpoints.astype(np.double)
        interior_cells = (interior_cells-.5).astype(np.double)
        nvertices, nfaces = compute_voromesh(vpoints, interior_cells)
        return nvertices, nfaces
    else:
        nvertices, faces, is_erronious = voronoi_to_poly_mesh(
            vpoints, interior_cells, clip)

        nfaces = []
        for face in faces:
            for i in range(2, len(face)):
                nfaces.append([face[0], face[i - 1], face[i]])

        if return_errors:
            return nvertices, np.array(nfaces), is_erronious
        else:
            return nvertices, np.array(nfaces)


def abs_max(t, t2):
    min = np.minimum(t, t2)
    max = np.maximum(t, t2)
    mask = np.abs(min) > np.abs(max)
    return min * mask + max * (1 - mask)


def get_clean_shape(V, sdf, mask_scale=2, grid_s=65):
    """V.values has to be in the [0 (outside), 1 (inside)] range (output of a sigmoid)
    V.points has to be in [-1, 1]^3"""
    vpoints = V.points.cpu().detach().numpy()
    pv = Voronoi(vpoints)
    interior_cells = (V.values.cpu().detach().numpy() - 0.5) > 0
    int_v = (pv.vertices + 1) / 2 * (grid_s - 1)
    int_v = np.clip(np.floor(int_v).astype(int), 0, grid_s - 2)
    pv_sdf = sdf[int_v[:, 0], int_v[:, 1], int_v[:, 2]]
    for ii in range(2):
        for ij in range(2):
            for ik in range(2):
                pv_sdf = abs_max(
                    pv_sdf, sdf[int_v[:, 0] + ii,
                                int_v[:, 1] + ij, int_v[:, 2] + ik]
                )
    warning = False
    for i, pr in enumerate(pv.point_region):
        reg_vertices = np.array(pv.regions[pr])
        if -1 in reg_vertices:
            interior_cells[i] = False
        else:
            r_sdf = pv_sdf[reg_vertices]
            is_out = (r_sdf > mask_scale * 2 / grid_s).sum()
            is_in = (r_sdf < -mask_scale * 2 / grid_s).sum()
            if is_out and is_in:
                warning = True
            elif is_out:
                interior_cells[i] = False
            elif is_in:
                interior_cells[i] = True
    if warning:
        print("warning: ambiguous region in this model")
    return voronoi_to_mesh(vpoints, interior_cells)


# model
class VoronoiValues(nn.Module):
    def __init__(self, points: np.ndarray, values: np.ndarray = None):
        super().__init__()
        if values is None:
            values = np.zeros_like(points[:, 0])
        self.values = nn.Parameter(
            torch.tensor(values, dtype=torch.float32), requires_grad=False
        )
        self.points = nn.Parameter(torch.tensor(points, dtype=torch.float32))
        self.kedge = 10  # search for nearest face
        self.knn = 10  # for normalization
        self.repulsion_fac = 100  # for repulsion

    # Cells addition/removal
    def replace_cells(self, points: np.ndarray, values: np.ndarray):
        """resets all cells"""
        rad_grad = self.values.requires_grad
        pt_grad = self.points.requires_grad
        with torch.no_grad():
            self.values = nn.Parameter(values, requires_grad=rad_grad)
            self.points = nn.Parameter(points, requires_grad=pt_grad)

    def subselect_cells(self, inds: np.ndarray):
        self.replace_cells(
            self.points[inds].detach(), self.values[inds].detach())

    def add_cells(self, points: np.ndarray, values: np.ndarray = None):
        if values is None:
            values = np.zeros_like(points[:, 0])
        device = self.points.device
        npoints = torch.cat(
            (self.points.detach(), torch.tensor(
                points, dtype=torch.float32).to(device))
        )
        nvalues = torch.cat(
            (self.values.detach(), torch.tensor(
                values, dtype=torch.float32).to(device))
        )
        self.replace_cells(npoints, nvalues)

    def new_random_cells(self):
        npoints = 2 * torch.rand_like(self.points) - 1
        nvalues = torch.zeros_like(self.values)
        self.replace_cells(npoints, nvalues)

    # Distance
    def voronoi_dist(self, points: torch.tensor):
        return ((points[:, None, :] - self.points[None, ...]) ** 2).sum(-1)

    def closest_cells(self, points: torch.tensor, number=2):
        return knn_points(points[None, :], self.points[None, :], K=number).idx[0]

    def squared_distance_to_edges(self, points: torch.tensor, return_indices=False):
        """computes distance to closest voronoi face WARNING: relying on knn"""
        indices = self.closest_cells(points, self.kedge + 1)
        point_to_voronoi_center = points - self.points[indices[:, 0]]
        voronoi_edge = self.points[indices[:, 1:]] - \
            self.points[indices[:, 0, None]]
        voronoi_edge_l = torch.sqrt(((voronoi_edge**2).sum(-1)))
        vector_length = (point_to_voronoi_center[:, None, :] * voronoi_edge).sum(
            -1
        ) / voronoi_edge_l
        sq_dist = (vector_length - voronoi_edge_l / 2) ** 2
        if return_indices:
            sind = indices[torch.arange(len(sq_dist)), 1 + sq_dist.min(1)[1]]
            return torch.hstack((indices[:, 0, None], sind[:, None]))
        return sq_dist.min(1)[0]

    def self_normalization(self):
        distance_to_voronoi = self.voronoi_dist(self.points)
        knn = min(self.knn, len(self.points) - 1)
        values, _ = distance_to_voronoi.topk(knn + 1, 1, False)
        return torch.exp(-self.repulsion_fac * values[:, 1:]) / self.repulsion_fac

    # Move cells
    def clamp(self):
        """clamps centers in bounding box"""
        with torch.no_grad():
            self.points[:] = torch.clamp(self.points, -1, 1)

    # Flag cells
    def select_relevant_cells(self, points: torch.tensor):
        """removes cells unactivated by the sample points"""
        with torch.no_grad():
            indices = self.squared_distance_to_edges(points, True)
            self.subselect_cells(torch.unique(indices))

    # Mesh
    def set_values_winding(self, v: np.ndarray, f: np.ndarray, barycenter=False):
        """sets values to winding number minus 0.5, from voronoi centers (default) or cell barycenters"""
        with torch.no_grad():
            if not (barycenter):
                self.values[:] = torch.tensor(
                    igl.fast_winding_number_for_meshes(
                        v, f, self.points.cpu().detach().numpy().astype("double")
                    )
                    - 0.5
                )

            else:
                vpoints = np.concatenate(
                    (self.points.cpu().detach().numpy(), SIGNS))
                pv = Voronoi(vpoints)
                vverts = []
                for e in pv.point_region:
                    delt = pv.vertices[pv.regions[e]]
                    vverts.append(delt.mean(0))
                self.vverts = np.row_stack(vverts)
                fws = igl.fast_winding_number_for_meshes(v, f, self.vverts)
                self.values[:] = torch.tensor(1.0 * (fws[:-8] > 0.5) - 0.5)

    def to_mesh(
        self, v: np.ndarray = None, f: np.ndarray = None, clip=True, return_errors=False
    ):
        """extracts mesh
            WARNING: the output mesh can contain self-intersections due to numerical errors. For a self-intersection free mesh, use the CGAL version in cpp_utils"""
        if not v is None:
            self.set_values_winding(v, f)
        return voronoi_to_mesh(
            self.points.cpu().detach().numpy(),
            self.values.cpu().detach().numpy() > 0,
            clip=clip,
            return_errors=return_errors,
        )

    def clean_useless_generators(self):
        pv = Voronoi(self.points.cpu().detach().numpy())
        interior = (self.values > 0).cpu().detach().numpy()
        keep = np.zeros(len(pv.points), dtype=bool)
        for e1, e2 in pv.ridge_points:
            if np.logical_xor(interior[e1], interior[e2]):
                keep[e1] = True
                keep[e2] = True
        self.subselect_cells(keep)


class VoronoiBaseNN(nn.Module):
    """to be used with neural networks"""

    def __init__(self):
        super().__init__()
        self.knn = 11

    def weighted_voronoi_dist(self, points, spoints):
        return ((points[:, :, None, :] - spoints[:, None, ...]) ** 2).sum(-1)

    def closest_cells(self, points, spoints, number):
        """points, self.points"""
        distance_to_voronoi = self.weighted_voronoi_dist(points, spoints)
        _, indices = distance_to_voronoi.topk(number, 2, False)
        return indices

    def squared_distance_to_edges(self, points, spoints):
        """samples of the groundtruth, voronoi centers"""
        # WARNING: fecthing for knn
        indices = self.closest_cells(points, spoints, self.knn)
        rindices = indices[..., None].repeat((1, 1, 1, 3))
        inside_cell = torch.gather(spoints, 1, rindices[:, :, 0, :])
        point_to_voronoi_center = points - inside_cell
        voronoi_edge = (
            torch.gather(
                spoints[:, :, None, :].repeat((1, 1, self.knn - 1, 3)),
                1,
                rindices[:, :, 1:, :],
            )
            - inside_cell[..., None, :]
        )
        voronoi_edge_l = torch.sqrt(((voronoi_edge**2).sum(-1)))
        vector_length = (point_to_voronoi_center[:, :, None, :] * voronoi_edge).sum(
            -1
        ) / voronoi_edge_l
        sq_dist = (vector_length - voronoi_edge_l / 2) ** 2
        return sq_dist.min(-1)[0]

    def weighted_voronoi_dist(self, points, spoints):
        return ((points.unsqueeze(2) - spoints.unsqueeze(1)) ** 2).sum(-1)

    def closest_cells(self, points, spoints, number):
        """points, self.points"""
        distance_to_voronoi = self.weighted_voronoi_dist(points, spoints)
        indices = torch.cat(
            [
                torch.topk(distance_to_voronoi[i], number, largest=False, sorted=True)[
                    1
                ].unsqueeze(0)
                for i in range(distance_to_voronoi.shape[0])
            ],
            dim=0,
        )
        return indices


class Voroloss(nn.Module):
    def __init__(self):
        super(Voroloss, self).__init__()
        self.knn = 11

    def weighted_voronoi_dist(self, points, spoints):
        return ((points.unsqueeze(2) - spoints.unsqueeze(1)) ** 2).sum(-1)

    def closest_cells(self, points, spoints, number):
        """points, self.points"""
        distance_to_voronoi = self.weighted_voronoi_dist(points, spoints)
        indices = torch.cat(
            [
                torch.topk(distance_to_voronoi[i], number, largest=False, sorted=True)[
                    1
                ].unsqueeze(0)
                for i in range(distance_to_voronoi.shape[0])
            ],
            dim=0,
        )
        return indices

    def __call__(self, points, spoints):
        """points, self.points"""
        # WARNING: fecthing for knn
        with torch.no_grad():
            indices = self.closest_cells(points, spoints, self.knn)
        inside_cell = spoints[
            np.kron(np.arange(spoints.shape[0]), np.ones(
                points.shape[1], dtype=int)),
            indices[:, :, 0].flatten(),
        ].reshape(spoints.shape[0], -1, 3)
        point_to_voronoi_center = points - inside_cell

        voronoi_edge = spoints[
            np.kron(
                np.arange(spoints.shape[0]),
                np.ones(points.shape[1] * (self.knn - 1), dtype=int),
            ),
            indices[:, :, 1:].flatten(),
        ].reshape(spoints.shape[0], -1, self.knn - 1, 3) - inside_cell.unsqueeze(2)

        voronoi_edge_l = torch.sqrt(((voronoi_edge**2).sum(-1)))
        vector_length = (point_to_voronoi_center[:, :, None, :] * voronoi_edge).sum(
            -1
        ) / voronoi_edge_l
        sq_dist = (vector_length - voronoi_edge_l / 2) ** 2
        return sq_dist.min(-1)[0]


class Voroloss_opt(nn.Module):
    def __init__(self):
        super(Voroloss_opt, self).__init__()
        self.knn = 16

    def __call__(self, points, spoints):
        """points, self.points"""
        # WARNING: fecthing for knn
        with torch.no_grad():
            indices = knn_points(points, spoints, K=self.knn).idx

        points_knn = knn_gather(spoints, indices)
        points_to_voronoi_center = points - points_knn[:, :, 0]

        voronoi_edge = points_knn[:, :, 1:] - points_knn[:, :, 0].unsqueeze(2)
        voronoi_edge_l = torch.sqrt(((voronoi_edge**2).sum(-1)))
        vector_length = (points_to_voronoi_center.unsqueeze(2) * voronoi_edge).sum(
            -1
        ) / voronoi_edge_l
        sq_dist = (vector_length - voronoi_edge_l / 2) ** 2
        return sq_dist.min(-1)[0]


class Voroloss_semi_opt(nn.Module):
    def __init__(self):
        super(Voroloss_semi_opt, self).__init__()
        self.knn = 11

    def __call__(self, points, spoints):
        """points, self.points"""
        # WARNING: fecthing for knn
        with torch.no_grad():
            indices = knn_points(points, spoints, K=self.knn).idx
        inside_cell = spoints[
            np.kron(np.arange(spoints.shape[0]), np.ones(
                points.shape[1], dtype=int)),
            indices[:, :, 0].flatten(),
        ].reshape(spoints.shape[0], -1, 3)
        point_to_voronoi_center = points - inside_cell

        voronoi_edge = spoints[
            np.kron(
                np.arange(spoints.shape[0]),
                np.ones(points.shape[1] * (self.knn - 1), dtype=int),
            ),
            indices[:, :, 1:].flatten(),
        ].reshape(spoints.shape[0], -1, self.knn - 1, 3) - inside_cell.unsqueeze(2)

        voronoi_edge_l = torch.sqrt(((voronoi_edge**2).sum(-1)))
        vector_length = (point_to_voronoi_center[:, :, None, :] * voronoi_edge).sum(
            -1
        ) / voronoi_edge_l
        sq_dist = (vector_length - voronoi_edge_l / 2) ** 2
        return sq_dist.min(-1)[0]


# Training
def train_voronoi(
    V: VoronoiValues,
    points: np.ndarray,
    optimizer: torch.optim.Optimizer,
    fac=1,
    clamp=True,
):
    optimizer.zero_grad()
    if fac < 1:
        mask = torch.rand_like(points[:, 0]) < fac
        points = points[mask]
    loss = V.squared_distance_to_edges(points).mean()
    if clamp:
        V.clamp()
    x = loss.item()
    loss.backward()
    optimizer.step()
    return x
