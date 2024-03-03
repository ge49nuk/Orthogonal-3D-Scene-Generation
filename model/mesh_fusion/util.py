import torch
import torchvision
import numpy as np
import open3d as o3d
import trimesh
import cv2
import copy

from poissonpy import functional, solvers
from scipy.optimize import minimize

from pytorch3d.renderer import FoVPerspectiveCameras, PerspectiveCameras
from pytorch3d.transforms import euler_angles_to_matrix


def get_pixel_grids(height, width, reverse=False):
    # texture coordinate.
    if reverse:
        # Pytorch3D expects +X left and +Y up!!!
        x_linspace = torch.linspace(width - 1, 0, width).view(1, width).expand(height, width)
        y_linspace = torch.linspace(height - 1, 0, height).view(height, 1).expand(height, width)
    else:
        x_linspace = torch.linspace(0, width - 1, width).view(1, width).expand(height, width)
        y_linspace = torch.linspace(0, height - 1, height).view(height, 1).expand(height, width)
    x_coordinates = x_linspace.contiguous().view(-1)
    y_coordinates = y_linspace.contiguous().view(-1)
    ones = torch.ones(height * width)
    indices_grid = torch.stack([x_coordinates,  y_coordinates, ones], dim=0)
    return indices_grid


def img_to_pts(height, width, depth, K=torch.eye(3)):
    pts = get_pixel_grids(height, width).to(depth.device)
    depth = depth.contiguous().view(1, -1)
    pts = pts * depth
    pts = torch.inverse(K).mm(pts)
    return pts


def pts_to_image(pts, K, RT):
    wrld_X = RT[:3, :3].mm(pts) + RT[:3, 3:4]
    xy_proj = K.mm(wrld_X)
    EPS = 1e-2
    mask = (xy_proj[2:3, :].abs() < EPS).detach()
    zs = xy_proj[2:3, :]
    zs[mask] = EPS
    sampler = torch.cat((xy_proj[0:2, :] / zs, xy_proj[2:3, :]), 0)

    # Remove invalid zs that cause nans
    sampler[mask.repeat(3, 1)] = -10
    return sampler, wrld_X


def Screen_to_NDC(x, H, W):
    sampler = torch.clone(x)
    sampler[0:1] = (sampler[0:1] / (W -1) * 2.0 -1.0) * (W - 1.0) / (H - 1.0)
    sampler[1:2] = (sampler[1:2] / (H -1) * 2.0 -1.0)
    return sampler


def get_camera(world_to_cam, fov_in_degrees):
    # pytorch3d expects transforms as row-vectors, so flip rotation: https://github.com/facebookresearch/pytorch3d/issues/1183
    R = world_to_cam[:3, :3].t()[None, ...]
    T = world_to_cam[:3, 3][None, ...]
    camera = FoVPerspectiveCameras(device=world_to_cam.device, R=R, T=T, fov=fov_in_degrees, degrees=True)
    #K = get_pinhole_intrinsics_from_fov(H, W, fov_in_degrees).to(world_to_cam.device)[None]
    #cameras = PerspectiveCameras(device=world_to_cam.device, R=R, T=T, in_ndc=False, K=K, image_size=torch.tensor([[H,W]]))
    return camera


def unproject_points(world_to_cam, fov_in_degrees, depth, H, W):
    camera = get_camera(world_to_cam, fov_in_degrees)

    xy_depth = get_pixel_grids(H, W, reverse=True).to(depth.device)
    xy_depth = Screen_to_NDC(xy_depth, H, W)
    xy_depth[2] = depth.flatten()
    xy_depth = xy_depth.T
    xy_depth = xy_depth[None]

    world_points = camera.unproject_points(xy_depth, world_coordinates=True, scaled_depth_input=False)
    #world_points = cameras.unproject_points(xy_depth, world_coordinates=True)
    world_points = world_points[0]
    world_points = world_points.T

    return world_points


def get_pinhole_intrinsics_from_fov(H, W, fov_in_degrees=55.0):
    px, py = (W - 1) / 2., (H - 1) / 2.
    fx = fy = W / (2. * np.tan(fov_in_degrees / 360. * np.pi))
    k_ref = np.array([[fx, 0.0, px, 0.0], [0.0, fy, py, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                     dtype=np.float32)
    k_ref = torch.tensor(k_ref)  # K is [4,4]

    return k_ref


def get_intrinsics(img, fov_in_degrees=55.0):
    C, H, W = img.shape
    if C != 3:
        H, W, C = img.shape
    k_ref = get_pinhole_intrinsics_from_fov(H, W, fov_in_degrees)
    if isinstance(img, torch.Tensor):
        k_ref = k_ref.to(img.device)

    return k_ref


def get_extrinsics(rot_xyz, trans_xyz, device="cpu"):
    T = torch.tensor(trans_xyz)
    R = euler_angles_to_matrix(torch.tensor(rot_xyz), "XYZ")

    RT = torch.cat([R, T[:, None]], dim=-1).to(device)  # RT is [4,4]
    RT = torch.cat([RT, torch.tensor([[0, 0, 0, 1]]).to(RT)], dim=0)

    return RT


def torch_to_o3d_mesh(vertices, faces, colors):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices.T.cpu().numpy())
    mesh.triangles = o3d.utility.Vector3iVector(faces.T.cpu().numpy())
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors.T.cpu().numpy())
    return mesh


def o3d_mesh_to_torch(mesh, v=None, f=None, c=None):
    vertices = torch.from_numpy(np.asarray(mesh.vertices)).T
    if v is not None:
        vertices = vertices.to(v)
    faces = torch.from_numpy(np.asarray(mesh.triangles)).T
    if f is not None:
        faces = faces.to(f)
    colors = torch.from_numpy(np.asarray(mesh.vertex_colors)).T
    if c is not None:
        colors = colors.to(c)
    return vertices, faces, colors


def torch_to_o3d_pcd(vertices, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices.T.cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(colors.T.cpu().numpy())
    return pcd


def o3d_pcd_to_torch(pcd, p=None, c=None):
    points = torch.from_numpy(np.asarray(pcd.points)).T
    if p is not None:
        points = points.to(p)
    colors = torch.from_numpy(np.asarray(pcd.colors)).T
    if c is not None:
        colors = colors.to(c)
    return points, colors


def torch_to_trimesh(vertices, faces, colors):
    mesh = trimesh.base.Trimesh(
        vertices=vertices.T.cpu().numpy(),
        faces=faces.T.cpu().numpy(),
        vertex_colors=(colors.T.cpu().numpy() * 255).astype(np.uint8),
        process=False)

    return mesh


def trimesh_to_torch(mesh: trimesh.base.Trimesh, v=None, f=None, c=None):
    vertices = torch.from_numpy(np.asarray(mesh.vertices)).T
    if v is not None:
        vertices = vertices.to(v)
    faces = torch.from_numpy(np.asarray(mesh.faces)).T
    if f is not None:
        faces = faces.to(f)
    colors = torch.from_numpy(np.asarray(mesh.visual.vertex_colors, dtype=float) / 255).T[:3]
    if c is not None:
        colors = colors.to(c)
    return vertices, faces, colors


def o3d_to_trimesh(mesh: o3d.geometry.TriangleMesh):
    return trimesh.base.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.triangles),
        vertex_colors=(np.asarray(mesh.vertex_colors).clip(0, 1) * 255).astype(np.uint8),
        process=False)

def NumpyToPCD(xyz):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd

def plane_pt_dist(x, plane, norm_p=None):
    if norm_p is None:
        norm_p = plane[:3]/np.linalg.norm(plane[:3])
    d = np.abs(np.matmul(x, norm_p) + plane[3]) 
    return d

def normalize_planes(planes):
    normals = planes[:,:3]
    normals = normals/np.tile(np.linalg.norm(normals,axis=1).reshape(-1,1), 3)
    planes[:,:3] = normals
    return planes

# Calculates the distances between given points
def calc_dists(pts):
    distances_full = []
    for pt1 in pts:
        distances_p1 = []
        for pt2 in pts:
            distances_p1.append(math.dist(pt1,pt2))
        distances_full.append(distances_p1)
    return np.array(distances_full)

def calc_angles(plane_eqs1, plane_eqs2):
    normals = plane_eqs1[:,:3]
    normals_T = plane_eqs2[:,:3].T
    n_dot = np.clip(np.dot(normals, normals_T), -1, 1)
    angles = np.arccos(n_dot) * (180/np.pi)
    return angles


def rotation_matrix_from_vectors(vec1, vec2):
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def Planar_patch_detection(points, colors, min_size):
    pcd = NumpyToPCD(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    oboxes = pcd.detect_planar_patches(
        normal_variance_threshold_deg=50,
        coplanarity_deg=70,
        outlier_ratio=0.75,
        min_plane_edge_length=0,
        min_num_points=500,
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=50))

    plane_eqs = np.array([])
    index = []
    for obox in oboxes:
        pts = obox.get_point_indices_within_bounding_box(pcd.points)
        if len(pts) < min_size:
            continue
        # calculate plane equation
        plane_pt = obox.get_center()
        plane_normal = obox.R[:,2]
        plane_eq = np.concatenate((plane_normal,[-np.dot(plane_pt,plane_normal)]))
        plane_eqs = np.append(plane_eqs, plane_eq)
        index.append(pts)
    return plane_eqs.reshape(-1,4), index

def Detect_Multi_Planes(points, colors, min_size=20000):
    plane_list = []
    plane_eqs, index = Planar_patch_detection(points, colors, min_size)
    for i,w in enumerate(plane_eqs):
        plane_list.append((w, index[i]))
    return plane_list

def Fit_points(orth_mode, plane_list, points, rot_mats, avg_depths, strictness=1.0):
    for i, tup in enumerate(plane_list): # [0]: plane eq. [1]:vertex coords [2]:vertex colors
        plane_eq, indices = tup
        pts = points[indices]
        # Project the new points onto orthogonalized planes
        if orth_mode == "proj":
            N = plane_eq[:3]
            dist = np.dot(pts,N) + plane_eq[3]
            pts_proj = pts - np.outer(strictness*dist,N)
            points[indices] = pts_proj
        # Rotate the plane with new vertices without projection°
        else:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.rotate(rot_mats[i])
            points[indices] = np.asarray(pcd.points)
    return avg_depths

def Orth_cost(plane_bias, plane_normals, verts):
    change_loss = 0
    for i, plane_verts in enumerate(verts):
        change_loss += np.median(plane_pt_dist(plane_verts, np.append(plane_normals[i],plane_bias[i]), plane_normals[i]))
    return change_loss

def Orthogonalize(orth_mode, plane_list, points):
    plane_list_new = plane_list
    ## DEFINE PARAMS FOR MINIMIZATION ##
    plane_verts = [points[plane[1]] for plane in plane_list]
    plane_eqs = np.array([item[0] for item in plane_list])
    ## ORTHOGONALIZE PLANE NORMALS ##
    rot_mats = []
    n1,n2,n3 = np.array([1,0,0]),np.array([0,1,0]),np.array([0,0,1])
    n4,n5,n6 = -n1,-n2,-n3
    orth_n = np.stack((n1,n2,n3,n4,n5,n6)).T
    for i,plane_eq in enumerate(plane_eqs):
        dot = np.dot(plane_eq[:3], orth_n)
        best_fit = np.argmax(dot)
        rot_mats.append(rotation_matrix_from_vectors(plane_eq[:3], orth_n.T[best_fit]))
        plane_eqs[i][:3] = orth_n.T[best_fit]
    if orth_mode == "proj": # not recommended
    ## RUN MINIMIZATION ##
        options = {"xrtol":0.0002}
        result = minimize(Orth_cost, plane_eqs[:,3].flatten(), args=(plane_eqs[:,:3], plane_verts), method='BFGS', options=options)
        plane_eqs[:,3] = result.x
    ## COMBINE VERTICES & EQUATIONS ##
        plane_list_new = []
        for i, tup in enumerate(plane_list):
            tup[0][:] = plane_eqs[i]
            plane_list_new.append(tup)
    return plane_list_new, rot_mats

def add_details(src_, tgt_):
# Modified version of https://github.com/bchao1/poissonpy/blob/master/examples/poisson_image_editing.py
    device = src_.device
    #normalize src and tgt
    src, tgt = copy.deepcopy(src_), copy.deepcopy(tgt_)
    max_dist_src, max_dist_tgt = torch.max(src).item(), torch.max(tgt).item()
    src /= max_dist_src
    tgt /= max_dist_tgt
    src = src.cpu().numpy()
    tgt = tgt.cpu().numpy()
    
    mask = np.ones_like(src)
    mask[:,0] = 0
    mask[:,-1] = 0
    mask[0] = 0
    mask[-1] = 0

    # compute laplacian of interpolation function
    Gx_src, Gy_src = functional.get_np_gradient(src)
    Gx_target, Gy_target = functional.get_np_gradient(tgt)
    G_src_mag = (Gx_src**2 + Gy_src**2)**0.5
    G_target_mag = (Gx_target**2 + Gy_target**2)**0.5
    Gx = np.where(G_src_mag > G_target_mag, Gx_src, Gx_target)
    Gy = np.where(G_src_mag > G_target_mag, Gy_src, Gy_target)
    Gxx, _ = functional.get_np_gradient(Gx, forward=False)
    _, Gyy = functional.get_np_gradient(Gy, forward=False)
    laplacian = Gxx + Gyy

    # solve interpolation function
    solver = solvers.Poisson2DRegion(mask, laplacian, tgt)
    solution = solver.solve()

    # alpha-blend interpolation and target function
    blended = mask * solution + (1 - mask) * tgt
    blended = np.clip(blended, 0, 1)
    blended *= max_dist_tgt
    blended = torch.from_numpy(blended).to(device)
    return blended

# This method replaces all zeros in a tensor with the closest(Manhattan) non-zero value
## WARNING: SLOW
def replace_zeros(tensor, device="cuda:0"):
    result_tensor = tensor.clone().cpu().numpy()
    zero_indices = np.argwhere(result_tensor == 0)
    rows, cols = zero_indices[:, 0], zero_indices[:, 1]
    non_zero_indices = np.argwhere(result_tensor != 0)
    non_zero_values = tensor[non_zero_indices[:, 0], non_zero_indices[:, 1]]
    for zero_row, zero_col in zip(rows, cols):
        if zero_row in [0,tensor.shape[0]-1] or zero_col in [0,tensor.shape[1]-1]:
            distances = np.abs(zero_row - non_zero_indices[:, 0]) + np.abs(zero_col - non_zero_indices[:, 1])
            closest_nonzero_index = np.argmin(distances)
            closest_nonzero_value = non_zero_values[closest_nonzero_index]
            result_tensor[zero_row, zero_col] = closest_nonzero_value
    return torch.tensor(result_tensor).to(device)