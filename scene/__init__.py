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

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers_4D import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON, load_cam_info
from tqdm import tqdm

from scene.dataset_4D import FourDGSdataset

from scene.gaussian_model4D import GaussianModel4D

class Scene:

    gaussians : GaussianModel4D

    def __init__(self, args : ModelParams, gaussians : GaussianModel4D, load_iteration=None, shuffle=True, resolution_scales=[1.0],
                  llffhold=8, override_train_idxs=None, override_test_idxs=None):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.video_cameras = {}

        print("source path: ", args.source_path)
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, 
                                                          llffhold=llffhold, override_train_idxs=override_train_idxs, override_test_idxs=override_test_idxs)
            dataset_type="colmap"
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
            print("Scene info keys: ", dir(scene_info))
            dataset_type="blender"
        else:
            assert False, "Could not recognize scene type!"

        self.dataset_type = dataset_type
        self.maxtime = scene_info.maxtime

        if not self.loaded_iter:
            # with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
            #     dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            if scene_info.video_cameras:
                camlist.extend(scene_info.video_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        num_views = len(scene_info.train_cameras)
        self.all_train_set = set(range(num_views))
        self.train_idxs = list(range(num_views))
        if shuffle:
            # Make this determinisjtic
            random.Random(42).shuffle(self.train_idxs)
            # The scene_info doesn't need to be shuffled 
            # random.Random(42).shuffle(scene_info.train_cameras)
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = FourDGSdataset(scene_info.train_cameras, args, dataset_type)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = FourDGSdataset(scene_info.test_cameras, args, dataset_type)
            print("Loading Video Cameras")
            self.video_cameras[resolution_scale] = FourDGSdataset(scene_info.video_cameras, args, dataset_type)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            # self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, self.maxtime)

            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, self.maxtime)
        
        self.candidate_views_filter = None

    def save(self, iteration, stage):
        if stage == "coarse":
            point_cloud_path = os.path.join(self.model_path, "point_cloud/coarse_iteration_{}".format(iteration))
        else:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_deformation(point_cloud_path)

    def getTrainCameras(self, scale=1.0):
        filted_train_camers = [self.train_cameras[scale][i] for i in self.train_idxs]
        return filted_train_camers

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    def getVideoCameras(self, scale = 1.0):
        return self.video_cameras[scale]
    
    def get_candidate_set(self):
        # Get candidate set 
        # Ensure resutls are always the same
        candidate_set = sorted(list(self.all_train_set - set(self.train_idxs)))
        if self.candidate_views_filter is not None:
            candidate_set = list(filter(self.candidate_views_filter, candidate_set))
        return candidate_set

    def getCandidateCameras(self, scale=1.0):
        candidate_set = list(self.get_candidate_set())
        filted_train_camers = [self.train_cameras[scale][i] for i in candidate_set]
        return filted_train_camers
    
    def load_inflated_cameras(self, info_path: str, inflate_skip: int):
        with open(info_path, "r") as f:
            info_dict = json.load(f)

        base_path = os.path.dirname(info_path)

        for scale in self.train_cameras.keys():
            assert scale == 1., "Didn't implement for other scale"
            base_idx = max(self.all_train_set) + 1

            load_idxs = [idx for idx in range(len(info_dict)) if idx % inflate_skip == 0]
            load_dicts = [info_dict[i] for i in load_idxs]

            inflated_cams = [load_cam_info(d, base_path=base_path) for d in tqdm(load_dicts, desc="Loading inflated cams")]

            self.all_train_set.update([base_idx + i for i, _ in enumerate(load_idxs)])
            self.train_cameras[scale].extend(inflated_cams)
