import numpy as np


def rotate_points_along_z(points, angle):
    """Rotation clockwise
    Args:
        points: np.array of np.array (B, N, 3 + C) or
            (N, 3 + C) for single batch
        angle: np.array of np.array (B, )
            or (, ) for single batch
            angle along z-axis, angle increases x ==> y
    Returns:
        points_rot:  (B, N, 3 + C) or (N, 3 + C)

    """
    single_batch = len(points.shape) == 2
    if single_batch:
        points = np.expand_dims(points, axis=0)
        angle = np.expand_dims(angle, axis=0)
    cosa = np.expand_dims(np.cos(angle), axis=1)
    sina = np.expand_dims(np.sin(angle), axis=1)
    zeros = np.zeros_like(cosa)  # angle.new_zeros(points.shape[0])
    ones = np.ones_like(sina)  # angle.new_ones(points.shape[0])

    rot_matrix = (
        np.concatenate((cosa, -sina, zeros, sina, cosa,
                       zeros, zeros, zeros, ones), axis=1)
        .reshape(-1, 3, 3)
    )

    # print(rot_matrix.view(3, 3))
    points_rot = np.matmul(points[:, :, :3], rot_matrix)
    points_rot = np.concatenate((points_rot, points[:, :, 3:]), axis=-1)

    if single_batch:
        points_rot = points_rot.squeeze(0)

    return points_rot


def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading],
            (x, y, z) is the box center

    Returns:
        corners: (N, 8, 3)
    """
    template = np.array([[1, 1, -1],
                         [1, -1, -1],
                         [-1, -1, -1],
                         [-1, 1, -1],
                         [1, 1, 1],
                         [1, -1, 1],
                         [-1, -1, 1],
                         [-1, 1, 1]]
                        ) / 2.

    # corners3d: of shape (N, 3, 8)
    corners3d = np.tile(boxes3d[:, None, 3:6],
                        (1, 8, 1)) * template[None, :, :]

    corners3d = rotate_points_along_z(corners3d.reshape(-1, 8, 3), boxes3d[:, 6]).reshape(
        -1, 8, 3
    )
    corners3d += boxes3d[:, None, 0:3]

    return corners3d


def points_in_boxes(points, boxes):
    """
    Args:
        pc: np.array (n, 3+d)
        boxes: np.array (m, 8, 3)
    Returns:
        mask: np.array (n, m) of type bool
    """
    if len(boxes) == 0:
        return np.zeros([points.shape[0], 1], dtype=np.bool)
    points = points[:, :3]  # get xyz
    # u = p6 - p5
    u = boxes[:, 6, :] - boxes[:, 5, :]  # (m, 3)
    # v = p6 - p7
    v = boxes[:, 6, :] - boxes[:, 7, :]  # (m, 3)
    # w = p6 - p2
    w = boxes[:, 6, :] - boxes[:, 2, :]  # (m, 3)

    # ux, vx, wx
    ux = np.matmul(points, u.T)  # (n, m)
    vx = np.matmul(points, v.T)
    wx = np.matmul(points, w.T)

    # up6, up5, vp6, vp7, wp6, wp2
    up6 = np.sum(u * boxes[:, 6, :], axis=1)
    up5 = np.sum(u * boxes[:, 5, :], axis=1)
    vp6 = np.sum(v * boxes[:, 6, :], axis=1)
    vp7 = np.sum(v * boxes[:, 7, :], axis=1)
    wp6 = np.sum(w * boxes[:, 6, :], axis=1)
    wp2 = np.sum(w * boxes[:, 2, :], axis=1)

    mask_u = np.logical_and(ux <= up6, ux >= up5)  # (1024, n)
    mask_v = np.logical_and(vx <= vp6, vx >= vp7)
    mask_w = np.logical_and(wx <= wp6, wx >= wp2)

    mask = mask_u & mask_v & mask_w  # (10240, n)

    return mask


def get_votes(points, gt_boxes):
    """
    Args:
        points: (N, 3)
        boxes: (m, 8, 3)
    Returns:
        votes: (N, 4)
    """
    n_point = points.shape[0]
    point_votes = np.zeros((n_point, 4)).astype(np.float32)
    for obj_id in range(gt_boxes.shape[0]):
        tmp_box3d = np.expand_dims(gt_boxes[obj_id], 0)  # (8, 3)
        # (n_point, 1)
        mask_pts = points_in_boxes(points[:, :3], tmp_box3d)
        mask_pts = mask_pts.reshape((-1,))
        point_votes[mask_pts, 0] = 1.0
        obj_center = np.mean(tmp_box3d, axis=1)  # (1, 3)

        # get votes
        pc_roi = points[mask_pts, :3]
        tmp_votes = obj_center - pc_roi
        point_votes[mask_pts, 1:4] = tmp_votes
    return point_votes
