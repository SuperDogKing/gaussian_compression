#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
from deepspeed.accelerator import get_accelerator
WARNED = False
def memory_usage():
    alloc = "mem_allocated: {:.4f} GB".format(get_accelerator().memory_allocated() / (1024 * 1024 * 1024))
    max_alloc = "max_mem_allocated: {:.4f} GB".format(get_accelerator().max_memory_allocated() /
                                                        (1024 * 1024 * 1024))
    cache = "cache_allocated: {:.4f} GB".format(get_accelerator().memory_cached() / (1024 * 1024 * 1024))
    max_cache = "max_cache_allocated: {:.4f} GB".format(get_accelerator().max_memory_cached() /
                                                        (1024 * 1024 * 1024))
    return " | {} | {} | {} | {}".format(alloc, max_alloc, cache, max_cache)

def loadCam(args, id, cam_info, resolution_scale, is_train=True):
    orig_w, orig_h = cam_info.image.size
    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w / (resolution_scale * args.resolution)), round(
            orig_h / (resolution_scale * args.resolution)
        )
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print(
                        "[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1"
                    )
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))
######################
    if is_train:
        resized_image_rgb = PILtoTorch(cam_info.image, resolution)
        gt_image = resized_image_rgb[:3, ...]
        loaded_mask = None
        if resized_image_rgb.shape[1] == 4:
            loaded_mask = resized_image_rgb[3:4, ...]
        return Camera(
            colmap_id=cam_info.uid,
            R=cam_info.R,
            T=cam_info.T,
            FoVx=cam_info.FovX,
            FoVy=cam_info.FovY,
            #########################
            image=gt_image,
            gt_alpha_mask=loaded_mask,
            ############################
            image_name=cam_info.image_name,
            uid=id,
            data_device=args.data_device,
        )
    else:
        # print("Before Camera",memory_usage())
        resized_image_rgb = PILtoTorch(cam_info.image, resolution)
        gt_image = resized_image_rgb[:3, ...]
        original_image = gt_image.clamp(0.0, 1.0)
        image_width = original_image.shape[2]
        image_height = original_image.shape[1]
        return Camera(
            colmap_id=cam_info.uid,
            R=cam_info.R,
            T=cam_info.T,
            FoVx=cam_info.FovX,
            FoVy=cam_info.FovY,
            image_width = image_width,
            image_height = image_height,
            #########################
            # image=gt_image,
            # gt_alpha_mask=loaded_mask,
            ############################
            image_name=cam_info.image_name,
            uid=id,
            data_device="cpu",
            # data_device=args.data_device,
        )


def cameraList_from_camInfos(cam_infos, resolution_scale, args, is_train=True):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale, is_train))

    return camera_list


def camera_to_JSON(id, camera: Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        "id": id,
        "img_name": camera.image_name,
        "width": camera.width,
        "height": camera.height,
        "position": pos.tolist(),
        "rotation": serializable_array_2d,
        "fy": fov2focal(camera.FovY, camera.height),
        "fx": fov2focal(camera.FovX, camera.width),
    }
    return camera_entry
