import networkx as nx
import numpy as np
import torch

newface_token = 0
stopface_token = 1
padface_token = 2


def get_shifted_sequence(sequence):
    non_special = np.flatnonzero(np.isin(sequence, [0, 1, 2], invert=True))
    if non_special.shape[0] > 0:
        idx = non_special[0]
        val = sequence[idx]
        sequence[non_special] -= val - 3
    return sequence


def read_faces(text):
    all_lines = text.splitlines()
    all_face_lines = [x for x in all_lines if x.startswith("f ")]
    all_faces = [
        [int(y.split("/")[0]) - 1 for y in x.strip().split(" ")[1:]]
        for x in all_face_lines
    ]
    return all_faces


def read_vertices(text):
    all_lines = text.splitlines()
    all_vertex_lines = [x for x in all_lines if x.startswith("v ")]
    all_vertices = np.array(
        [[float(y) for y in x.strip().split(" ")[1:]] for x in all_vertex_lines]
    )
    assert all_vertices.shape[1] == 3, "vertices should have 3 coordinates"
    return all_vertices


def quantize_coordinates(coords, num_tokens=256):
    if torch.is_tensor(coords):
        coords = torch.clip((coords + 0.5), 0, 1) * num_tokens  # type: ignore
        coords_quantized = coords.round().long()
    else:
        coords = np.clip((coords + 0.5), 0, 1) * num_tokens  # type: ignore
        coords_quantized = coords.round().astype(int)
    return coords_quantized


def face_to_cycles(face):
    """Find cycles in face."""
    g = nx.Graph()
    for v in range(len(face) - 1):
        g.add_edge(face[v], face[v + 1])
    g.add_edge(face[-1], face[0])
    return list(nx.cycle_basis(g))


def sort_vertices_and_faces(vertices_, faces_, num_tokens=256):
    vertices = np.clip((vertices_ + 0.5), 0, 1) * num_tokens  # type: ignore
    vertices_quantized_ = vertices.round().astype(int)

    vertices_quantized_ = vertices_quantized_[:, [2, 0, 1]]
    vertices_quantized, unique_inverse = np.unique(
        vertices_quantized_, axis=0, return_inverse=True
    )

    sort_inds = np.lexsort(vertices_quantized.T)

    vertices_quantized = vertices_quantized[sort_inds]
    vertices_quantized = np.stack(
        [vertices_quantized[:, 2], vertices_quantized[:, 1], vertices_quantized[:, 0]],
        axis=-1,
    )

    # Re-index faces and tris to re-ordered vertices.
    faces = [np.argsort(sort_inds)[unique_inverse[f]] for f in faces_]
    # Merging duplicate vertices and re-indexing the faces causes some faces to
    # contain loops (e.g [2, 3, 5, 2, 4]). Split these faces into distinct
    # sub-faces.
    sub_faces = []
    for f in faces:
        cliques = face_to_cycles(f)
        for c in cliques:
            c_length = len(c)
            # Only append faces with more than two verts.
            if c_length > 2:
                d = np.argmin(c)
                # Cyclically permute faces just that first index is the smallest.
                sub_faces.append([c[(d + i) % c_length] for i in range(c_length)])
    faces = sub_faces
    # Sort faces by lowest vertex indices. If two faces have the same lowest
    # index then sort by next lowest and so on.
    faces.sort(key=lambda f: tuple(sorted(f)))

    # After removing degenerate faces some vertices are now unreferenced.
    # Remove these.
    num_verts = vertices_quantized.shape[0]
    try:
        vert_connected = np.equal(
            np.arange(num_verts)[:, None], np.hstack(faces)[None]
        ).any(axis=-1)
    except:
        return None, None
    vertices_quantized = vertices_quantized[vert_connected]
    # Re-index faces and tris to re-ordered vertices.
    vert_indices = np.arange(num_verts) - np.cumsum(1 - vert_connected.astype("int"))
    faces = [vert_indices[f].tolist() for f in faces]
    vertices = vertices_quantized / num_tokens - 0.5
    # order: Z, Y, X --> X, Y, Z
    vertices = np.stack([vertices[:, 2], vertices[:, 1], vertices[:, 0]], axis=-1)
    return vertices, faces


def sort_vertices_and_faces_wpoints(vertices_, faces_, points, num_tokens=256):
    vertices = np.clip((vertices_ + 0.5), 0, 1) * num_tokens  # type: ignore
    points = np.clip((points + 0.5), 0, 1) * num_tokens  # type: ignore

    vertices_quantized_ = vertices.round().astype(int)
    points_quantized_ = points.round().astype(int)

    vertices_quantized_ = vertices_quantized_[:, [2, 0, 1]]
    points_quantized_ = points_quantized_[:, [2, 0, 1]]
    vertices_quantized, unique_inverse = np.unique(
        vertices_quantized_, axis=0, return_inverse=True
    )

    sort_inds = np.lexsort(vertices_quantized.T)

    vertices_quantized = vertices_quantized[sort_inds]
    vertices_quantized = np.stack(
        [vertices_quantized[:, 2], vertices_quantized[:, 1], vertices_quantized[:, 0]],
        axis=-1,
    )
    points_quantized = np.stack(
        [points_quantized_[:, 2], points_quantized_[:, 1], points_quantized_[:, 0]],
        axis=-1,
    )

    # Re-index faces and tris to re-ordered vertices.
    faces = [np.argsort(sort_inds)[unique_inverse[f]] for f in faces_]
    # Merging duplicate vertices and re-indexing the faces causes some faces to
    # contain loops (e.g [2, 3, 5, 2, 4]). Split these faces into distinct
    # sub-faces.
    sub_faces = []
    for f in faces:
        cliques = face_to_cycles(f)
        for c in cliques:
            c_length = len(c)
            # Only append faces with more than two verts.
            if c_length > 2:
                d = np.argmin(c)
                # Cyclically permute faces just that first index is the smallest.
                sub_faces.append([c[(d + i) % c_length] for i in range(c_length)])
    faces = sub_faces
    # Sort faces by lowest vertex indices. If two faces have the same lowest
    # index then sort by next lowest and so on.
    faces.sort(key=lambda f: tuple(sorted(f)))

    # After removing degenerate faces some vertices are now unreferenced.
    # Remove these.
    num_verts = vertices_quantized.shape[0]
    vert_connected = np.equal(
        np.arange(num_verts)[:, None], np.hstack(faces)[None]
    ).any(axis=-1)
    vertices_quantized = vertices_quantized[vert_connected]
    # Re-index faces and tris to re-ordered vertices.
    vert_indices = np.arange(num_verts) - np.cumsum(1 - vert_connected.astype("int"))
    faces = [vert_indices[f].tolist() for f in faces]
    vertices = vertices_quantized / num_tokens - 0.5
    points = points_quantized / num_tokens - 0.5
    # order: Z, Y, X --> X, Y, Z
    vertices = np.stack([vertices[:, 2], vertices[:, 1], vertices[:, 0]], axis=-1)
    points = np.stack([points[:, 2], points[:, 1], points[:, 0]], axis=-1)

    return vertices, faces, points


def fill_and_perm(faces, face_neighborhood, features, num_faces=100):
    """
    Fill the faces and features toward using NumPy operations.
    """
    Nf, C = features.shape
    assert len(faces) == Nf

    faces = np.array(faces, dtype=np.int64)

    pad_count = num_faces - Nf
    if pad_count > 0:
        pad_faces = np.full((pad_count, 3), -1, dtype=np.int64)
        pad_feats = np.full((pad_count, C), -1.0, dtype=features.dtype)
        faces = np.concatenate([faces, pad_faces], axis=0)
        features = np.concatenate([features, pad_feats], axis=0)

    features_ori = features.copy()
    faces_ori = faces.copy()

    # Create a mask to indicate which faces are padded
    mask = np.zeros(num_faces, dtype=bool)
    mask[:Nf] = True
    mask_ori = mask.copy()

    # Permute
    perm = np.random.permutation(num_faces)

    # try:
    #     faces_permuted = [faces[i] for i in perm]
    #     features_permuted = features[perm]
    #     mask_permuted = mask[perm]
    # except:
    #     a = 1

    faces_permuted = faces[perm]
    features_permuted = features[perm]
    mask_permuted = mask[perm]

    # Convert face_neighborhood to a NumPy array for easier manipulation
    face_neighborhood = np.array(face_neighborhood)

    # Initialize a new face neighborhood array
    face_neighborhood_permuted = face_neighborhood.copy()

    # Adjust the face neighborhood
    for i in range(face_neighborhood.shape[0]):
        for j in range(face_neighborhood.shape[1]):
            # Find the new index for each face in the neighborhood after permutation
            old_face_index = face_neighborhood[i, j]
            new_face_index = np.where(perm == old_face_index)[0][0]
            face_neighborhood_permuted[i, j] = new_face_index

    return (
        faces_permuted,
        faces_ori,
        face_neighborhood_permuted,
        features_permuted,
        features_ori,
        mask_permuted,
        mask_ori,
    )


def sort_vertices_and_faces_quads(vertices_, faces_, num_tokens=256):
    vertices = np.clip((vertices_ + 0.5), 0, 1) * num_tokens  # type: ignore
    vertices_quantized_ = vertices.round().astype(int)

    # vertices_quantized_ = vertices_quantized_[:, [2, 0, 1]]
    vertices_quantized, unique_inverse = np.unique(
        vertices_quantized_, axis=0, return_inverse=True
    )

    #
    sort_inds = np.lexsort(vertices_quantized.T)

    vertices_quantized = vertices_quantized[sort_inds]
    # vertices_quantized = np.stack(
    #     [vertices_quantized[:, 2], vertices_quantized[:, 1], vertices_quantized[:, 0]],
    #     axis=-1,
    # )

    # Re-index faces and tris to re-ordered vertices.
    faces = [np.argsort(sort_inds)[unique_inverse[f]] for f in faces_]
    # Merging duplicate vertices and re-indexing the faces causes some faces to
    # contain loops (e.g [2, 3, 5, 2, 4]). Split these faces into distinct
    # sub-faces.
    sub_faces = []
    for f in faces:
        cliques = face_to_cycles(f)
        for c in cliques:
            c_length = len(c)
            # Only append faces with more than two verts.
            if c_length > 2:
                d = np.argmin(c)
                # Cyclically permute faces just that first index is the smallest.
                sub_faces.append([c[(d + i) % c_length] for i in range(c_length)])
    faces = sub_faces
    # Sort faces by lowest vertex indices. If two faces have the same lowest
    # index then sort by next lowest and so on.
    faces.sort(key=lambda f: tuple(sorted(f)))

    # After removing degenerate faces some vertices are now unreferenced.
    # Remove these.
    num_verts = vertices_quantized.shape[0]
    vert_connected = np.equal(
        np.arange(num_verts)[:, None], np.hstack(faces)[None]
    ).any(axis=-1)
    vertices_quantized = vertices_quantized[vert_connected]
    # Re-index faces and tris to re-ordered vertices.
    vert_indices = np.arange(num_verts) - np.cumsum(1 - vert_connected.astype("int"))
    faces = [vert_indices[f].tolist() for f in faces]
    vertices = vertices_quantized / num_tokens - 0.5
    # order: Z, Y, X --> X, Y, Z
    # vertices = np.stack([vertices[:, 2], vertices[:, 1], vertices[:, 0]], axis=-1)
    return vertices, faces
