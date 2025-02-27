import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Union, Optional, Tuple
from copy import deepcopy
import random
from gaussian_renderer import render, network_gui, modified_render
from scene import Scene
from collections import OrderedDict


class HRegSelector(torch.nn.Module):

    def __init__(self, args) -> None:
        super().__init__()
        self.seed = args.seed
        self.reg_lambda = args.reg_lambda
        self.I_test: bool = args.I_test
        self.I_acq_reg: bool = args.I_acq_reg

        name2idx = {"xyz": 0, "rgb": 1, "sh": 2, "scale": 3, "rotation": 4, "opacity": 5}
        self.filter_out_idx: List[str] = [name2idx[k] for k in args.filter_out_grad]

    
    def nbvs(self, gaussians, scene: Scene, num_views, pipe, background, exit_func) -> List[int]:
        candidate_views = list(deepcopy(scene.get_candidate_set()))

        viewpoint_cams = scene.getTrainCameras().copy()

        if self.I_test == True:
            viewpoint_cams = scene.getTestCameras()

        params = gaussians.capture()[1:7]

        params = [p for i, p in enumerate(params) if i not in self.filter_out_idx]


        # off load to cpu to avoid oom with greedy algo
        device = params[0].device if num_views == 1 else "cpu"
        # device = "cpu" # we have to load to cpu because of inflation

        print("Filter Out IDX: ", self.filter_out_idx)
        # for p in params:
        #     if isinstance(p, OrderedDict):
        #         print("p is an OrderedDict")
        #         print("P: ", p)
        #     else:
        #         print("p is not an OrderedDict")
        #         if p.grad is not None:  # Ensure the gradient exists
        #             if torch.isnan(p.grad).any():
        #                 print("p has a NaN gradient.")
        #             else:
        #                 print("p does not have a NaN gradient.")
        #         else:
        #             print("p has no gradient.")

        H_train = torch.zeros(sum(p.numel() for p in params), device=params[0].device, dtype=params[0].dtype)

        candidate_cameras = scene.getCandidateCameras()
        # Run heesian on training set
        print("Viewpoint Cams: ", viewpoint_cams)
        for cam in tqdm(viewpoint_cams, desc="Calculating diagonal Hessian on training views"):
            if exit_func():
                raise RuntimeError("csm should exit early")

            render_pkg = modified_render(cam, gaussians, pipe, background)
            pred_img = render_pkg["render"]

            print("Predicted image contains NaN: ", torch.isnan(pred_img).any())
            print("Predicted image contains Inf: ", torch.isinf(pred_img).any())
            print("Predicted image shape: ", pred_img.shape)
            print("")

            pred_img.backward(gradient=torch.ones_like(pred_img))

            # for p in params:
            #     print(f"Gradient for parameter {p}: {p.grad}")
            cur_H = torch.cat([p.grad.detach().reshape(-1) for p in params])

            print("Cur_H contains NaN in Training views: ", torch.isnan(cur_H).any())

            H_train += cur_H

            gaussians.optimizer.zero_grad(set_to_none = True) 

        H_train = H_train.to(device)
        if num_views == 1:
            print("Num views is 1")
            return self.select_single_view(H_train, candidate_cameras, candidate_views, gaussians, pipe, background, params, exit_func)

        H_candidates = []
        for idx, cam in enumerate(tqdm(candidate_cameras, desc="Calculating diagonal Hessian on candidate views")):
            if exit_func():
                raise RuntimeError("csm should exit early")

            render_pkg = modified_render(cam, gaussians, pipe, background)
            pred_img = render_pkg["render"]
            pred_img.backward(gradient=torch.ones_like(pred_img))

            cur_H = torch.cat([p.grad.detach().reshape(-1) for p in params])

            H_candidates.append(cur_H.to(device))

            gaussians.optimizer.zero_grad(set_to_none = True) 
        
        selected_idxs = []

        for _ in range(num_views):
            I_train = torch.reciprocal(H_train + self.reg_lambda)
            if self.I_acq_reg:
                acq_scores = np.array([torch.sum((cur_H + self.I_acq_reg) * I_train).item() for cur_H in H_candidates])
            else:
                acq_scores = np.array([torch.sum(cur_H * I_train).item() for cur_H in H_candidates])
            selected_idx = acq_scores.argmax()
            selected_idxs.append(candidate_views.pop(selected_idx))

            H_train += H_candidates.pop(selected_idx)

        return selected_idxs

    
    def forward(self, x):
        return x
    
    
    def select_single_view(self, I_train, candidate_cameras, candidate_views, gaussians, pipe, background, params, exit_func, num_views=1):
        """
        A memory effcient way when doing single view selection
        """
        I_train = torch.reciprocal(I_train + self.reg_lambda)
        acq_scores = torch.zeros(len(candidate_cameras))
        for idx, cam in enumerate(tqdm(candidate_cameras, desc="Calculating diagonal Hessian on candidate views")):
            if exit_func():
                raise RuntimeError("csm should exit early")
            print("Rendeirng camera: ", cam)
            render_pkg = modified_render(cam, gaussians, pipe, background)
            print("rendered successfully")
            pred_img = render_pkg["render"]
            
            print("Backward")
            pred_img.backward(gradient=torch.ones_like(pred_img))

            print("Concat")
            cur_H = torch.cat([p.grad.detach().reshape(-1) for p in params])

            I_acq = cur_H

            print("Cur_H contains NaN: ", torch.isnan(cur_H).any())

            if self.I_acq_reg:
                I_acq += self.reg_lambda

            print("zero grad")
            gaussians.optimizer.zero_grad(set_to_none = True) 

            print("sum")
            print("I_acq is CUDA:", I_acq.is_cuda)
            print("I_acq shape:", I_acq.shape)
            print("I_acq dtype:", I_acq.dtype)
            print("I_acq contains NaN:", torch.isnan(I_acq).any())
            # print("I_acq contains Inf:", torch.isinf(I_acq).any())
            print("I_train shape:", I_train.shape)
            print("I_train device:", I_train.device)
            print("self.reg_lambda:", self.reg_lambda)
            print("self.reg_lambda device:", self.reg_lambda.device if isinstance(self.reg_lambda, torch.Tensor) else "scalar")
            acq_scores[idx] += torch.sum(I_acq * I_train).item()

            print("Summed")
        
        print(f"acq_scores: {acq_scores.tolist()}")
        if self.I_test == True:
            acq_scores *= -1

        _, indices = torch.sort(acq_scores, descending=True)
        selected_idxs = [candidate_views[i] for i in indices[:num_views].tolist()]
        return selected_idxs
