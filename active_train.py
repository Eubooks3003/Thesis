import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui, modified_render
import sys
from scene import Scene, GaussianModel4D
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from active.schema import schema_dict
from utils.loss_utils import ssim
from lpipsPyTorch import lpips, lpips_func
from active import methods_dict
import wandb
import copy
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
from utils.cluster_manager import ClusterStateManager

csm = ClusterStateManager()

@torch.no_grad()
def save_checkpoint(gaussians, iteration, scene, base_iter=0, save_path=None, save_last=True):
    ckpt_dict = {"model_params": gaussians.capture(), "first_iter": iteration, "train_idx": scene.train_idxs, "base_iter": base_iter}

    if save_last:
        last_path = scene.model_path + "/last.pth"
        print("\n[ITER {}] Saving Checkpoint to {}".format(iteration, last_path))
        torch.save(ckpt_dict, last_path)   

    if save_path is None:
        save_path = scene.model_path + "/chkpnt" + str(iteration) + ".pth"
    print("\n[ITER {}] Saving Checkpoint to {}".format(iteration, save_path))
    torch.save(ckpt_dict, save_path)   

def load_checkpoint(ckpt_path: str, gaussians, scene, opt, ignore_train_idxs=False):
    ckpt_dict = torch.load(ckpt_path)
    (model_params, first_iter, train_idxs) = ckpt_dict["model_params"], ckpt_dict["first_iter"], ckpt_dict["train_idx"]
    gaussians.restore(model_params, opt)
    if not ignore_train_idxs:
        scene.train_idxs = train_idxs

    base_iter = ckpt_dict.get("base_iter", 0)
    return first_iter, base_iter

def training(dataset, hyper, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args):
    first_iter = 0
    base_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    # Use Gaussian Model from 4D Gaussian
    gaussians = GaussianModel4D(dataset.sh_degree, hyper)

    # Use Scene from Fisher RF
    scene = Scene(dataset, gaussians)

    schema = schema_dict[args.schema](dataset_size=len(scene.getTrainCameras()), scene=scene)
    print(f"schema: {schema.load_its}")
    scene.train_idxs = schema.init_views

    active_method = methods_dict[args.method](args)

    print("Coarse Iterations: ", opt.coarse_iterations)
    print("Iterations: ",  opt.iterations)
    scene_reconstruction_4D(dataset, hyper, opt, pipe, testing_iterations, saving_iterations,
                            checkpoint_iterations, checkpoint, debug_from,
                            gaussians, scene, "coarse", tb_writer, opt.coarse_iterations, active_method, schema)

    op_iterations = 30_000 # TODO: Figure out what is happening why is it 20_000????
    scene_reconstruction_4D(dataset, hyper, opt, pipe, testing_iterations, saving_iterations,
                        checkpoint_iterations, checkpoint, debug_from,
                        gaussians, scene, "fine", tb_writer, op_iterations, active_method, schema)                      
    


def scene_reconstruction(dataset, hyper, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from,
gaussians, scene, stage, tb_writer, train_iter):
    first_iter = 0
    base_iter = 0
    # tb_writer = prepare_output_and_logger(dataset)

    # # Use Gaussian Model from 4D Gaussian
    # gaussians = GaussianModel4D(dataset.sh_degree, hyper)

    # Use Scene from Fisher RF
    # scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    
    # Active View Selection
    schema = schema_dict[args.schema](dataset_size=len(scene.getTrainCameras()), scene=scene)
    print(f"schema: {schema.load_its}")

    if False:
        scene.train_idxs = schema.init_views

        # TODO: Temporary hack, fix so that the views get added throughout corase + fine

    active_method = methods_dict[args.method](args)

    init_ckpt_path = f"{args.model_path}/init.ckpt"
    if checkpoint: # this is to continue training in SLURM after requeue
        if os.path.exists(checkpoint):
            # Check this for 4D Gaussian
            first_iter, base_iter = load_checkpoint(checkpoint, gaussians, scene, opt)
        else:
            print(f"[WARNING] checkpoint {checkpoint} doesn't exist, training from scratch")

    if first_iter == 0: # maybe init_ckpt has been save if preempted
        save_checkpoint(gaussians, first_iter, scene, base_iter, save_path=init_ckpt_path, save_last=False)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    print("Iterations: ", opt.iterations)
    progress_bar = tqdm(range(first_iter, train_iter), desc="Training progress")
    first_iter += 1

    batch_size = opt.batch_size
    print("data loading done")
    if opt.dataloader:
        viewpoint_stack = scene.getTrainCameras()
        if opt.custom_sampler is not None:
            sampler = FineSampler(viewpoint_stack)
            viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size,sampler=sampler,num_workers=16,collate_fn=list)
            random_loader = False
        else:
            viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size,shuffle=True,num_workers=16,collate_fn=list)
            random_loader = True
        loader = iter(viewpoint_stack_loader)
    

    if stage == "coarse" and opt.zerostamp_init:
        load_in_memory = True
        # batch_size = 4
        temp_list = get_stamp_list(viewpoint_stack,0)
        viewpoint_stack = temp_list.copy()
    else:
        load_in_memory = False 

    for iteration in range(first_iter, train_iter + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    # Use FisherRF render
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(train_iter)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None
        
        if False:
            num_views = schema.num_views_to_add(iteration)
        else:
            num_views = 0
        # TODO: Temporary hack, fix so that the views get added throughout corase + fine
       
        if num_views > 0:
            try:
                # For sectioned training
                candidate_views_filter = getattr(schema, "candidate_views_filter")[iteration] if hasattr(schema, "candidate_views_filter") else None
                scene.candidate_views_filter = candidate_views_filter
                
                # Because selection is time consumeing
                selected_views = active_method.nbvs(gaussians, scene, num_views, pipe, background, exit_func=csm.should_exit)
            except RuntimeError as e:
                print(e)
                print("selector exited early")
                # NOTE: we use iteration - 1 because the selector is not done
                save_checkpoint(gaussians, iteration - 1, scene)
                csm.requeue()

            print(f"ITER {iteration}: selected views: {selected_views}")
            scene.train_idxs.extend(selected_views)
            print(f"ITER {iteration}: training views after selection: {scene.train_idxs}")

            gaussians.optimizer.zero_grad(set_to_none = True)
            first_iter, _ = load_checkpoint(init_ckpt_path, gaussians, scene, opt, ignore_train_idxs=True)
            print("Loaded From Init checkpoint, gaussians state_dict: ",  gaussians._deformation.deformation_net.grid.grids)
            base_iter = iteration - 1

        iter_start.record()

        gaussians.update_learning_rate(iteration - base_iter)

        if iteration > args.sh_up_after and iteration % args.sh_up_every == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        batch_size=opt.batch_size
        if opt.dataloader and not load_in_memory:
            try:
                viewpoint_cams = next(loader)
            except StopIteration:
                print("reset dataloader into random dataloader.")
                if not random_loader:
                    viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=opt.batch_size,shuffle=True,num_workers=32,collate_fn=list)
                    random_loader = True
                loader = iter(viewpoint_stack_loader)
        
        else:
            idx = 0
            viewpoint_cams = []

            while idx < batch_size :    
                    
                viewpoint_cam = viewpoint_stack.pop(randint(0,len(viewpoint_stack)-1))
                viewpoint_cams.append(viewpoint_cam)
                idx +=1
            if len(viewpoint_cams) == 0:
                continue

        # viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        images = []
        gt_images = []
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []
        for viewpoint_cam in viewpoint_cams:
            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            images.append(image.unsqueeze(0))
            gt_image = viewpoint_cam.original_image.cuda()
            
            gt_images.append(gt_image.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)
        
        # render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        # image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        radii = torch.cat(radii_list,0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        image_tensor = torch.cat(images,0)
        gt_image_tensor = torch.cat(gt_images,0)

        # Loss

        # Fisher RF stuff
        # gt_image = viewpoint_cam.original_image.cuda()
        # Ll1 = l1_loss(image, gt_image)
        # loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        Ll1 = l1_loss(image_tensor, gt_image_tensor[:,:3,:,:])
        psnr_ = psnr(image_tensor, gt_image_tensor).mean().double()

        loss = Ll1
        if stage == "fine" and hyper.time_smoothness_weight != 0:
            # tv_loss = 0
            tv_loss = gaussians.compute_regulation(hyper.time_smoothness_weight, hyper.l1_time_planes, hyper.plane_tv_weight)
            loss += tv_loss
        if opt.lambda_dssim != 0:
            ssim_loss = ssim(image_tensor,gt_image_tensor)
            loss += opt.lambda_dssim * (1.0-ssim_loss)

        loss.backward()

        if torch.isnan(loss).any():
            print("loss is nan,end training, reexecv program now.")
            os.execv(sys.executable, [sys.executable] + sys.argv)
        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        for idx in range(0, len(viewspace_point_tensor_list)):
            viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad

        iter_end.record()

        # We save before logging
        if csm.should_exit():
            save_checkpoint(gaussians, iteration - 1, scene)
            csm.requeue()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                                          "psnr": f"{psnr_:.{2}f}"})
                progress_bar.update(10)
            if iteration == train_iter:
                progress_bar.close()

            before_selection = schema.num_views_to_add(iteration + 1) > 0
            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), 
                            testing_iterations, scene, render, (pipe, background), before_selection=before_selection, 
                            log_every_image=args.log_every_image)

            # stage = "coarse" # Only use fine for now until I figure out how to get the coarse and fine to interface with FisherRF
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                print("Saving Gaussians:  ",  gaussians._deformation.deformation_net.grid.grids)
                scene.save(iteration, stage)

            # Densification
            cur_iter = iteration - base_iter
            if cur_iter < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if stage == "coarse":
                    opacity_threshold = opt.opacity_threshold_coarse
                    densify_threshold = opt.densify_grad_threshold_coarse
                else:    
                    opacity_threshold = opt.opacity_threshold_fine_init - cur_iter*(opt.opacity_threshold_fine_init - opt.opacity_threshold_fine_after)/(opt.densify_until_iter)  
                    densify_threshold = opt.densify_grad_threshold_fine_init - cur_iter*(opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_after)/(opt.densify_until_iter )  
                if  cur_iter > opt.densify_from_iter and cur_iter % opt.densification_interval == 0 and gaussians.get_xyz.shape[0]<360000:
                    size_threshold = 20 if cur_iter > opt.opacity_reset_interval else None
                    
                    gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold, 5, 5, scene.model_path, cur_iter, stage)
                if  cur_iter > opt.pruning_from_iter and cur_iter % opt.pruning_interval == 0 and gaussians.get_xyz.shape[0]>200000:
                    size_threshold = 20 if cur_iter > opt.opacity_reset_interval else None

                    gaussians.prune(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                    
                # if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 :
                if cur_iter % opt.densification_interval == 0 and gaussians.get_xyz.shape[0]<360000 and opt.add_point:
                    gaussians.grow(5,5,scene.model_path,cur_iter,stage)
                    # torch.cuda.empty_cache()
                if cur_iter % opt.opacity_reset_interval == 0:
                    print("reset opacity")
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < train_iter:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
        
        if (iteration in checkpoint_iterations):
            print("Saving: ", gaussians._deformation.grid.grids)
            save_checkpoint(gaussians, iteration, scene)
    wandb.finish()


def scene_reconstruction_4D(dataset, hyper, opt, pipe, testing_iterations, saving_iterations, 
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, stage, tb_writer, train_iter, active_method, schema):
    first_iter = 0

    gaussians.training_setup(opt)
    if checkpoint:
        # breakpoint()
        if stage == "coarse" and stage not in checkpoint:
            print("start from fine stage, skip coarse stage.")
            # process is in the coarse stage, but start from fine stage
            return
        if stage in checkpoint: 
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)


    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0

    final_iter = train_iter
    
    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1
    # lpips_model = lpips.LPIPS(net="alex").cuda()
    video_cams = scene.getVideoCameras()
    test_cams = scene.getTestCameras()
    train_cams = scene.getTrainCameras()

    # Comment out since active view selection should start with very little views
    # if not viewpoint_stack and not opt.dataloader:
    #     # dnerf's branch
    #     viewpoint_stack = [i for i in train_cams]
    #     temp_list = copy.deepcopy(viewpoint_stack)

    # 
    batch_size = opt.batch_size
    print("data loading done")

    # if opt.dataloader:
    #     viewpoint_stack = scene.getTrainCameras()
    #     if opt.custom_sampler is not None:
    #         sampler = FineSampler(viewpoint_stack)
    #         viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size,sampler=sampler,num_workers=16,collate_fn=list)
    #         random_loader = False
    #     else:
    #         viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size,shuffle=True,num_workers=16,collate_fn=list)
    #         random_loader = True
    #     loader = iter(viewpoint_stack_loader)
    
    
    # dynerf, zerostamp_init
    # breakpoint()
    if stage == "coarse" and opt.zerostamp_init:
        load_in_memory = True
        # batch_size = 4
        temp_list = get_stamp_list(viewpoint_stack,0)
        viewpoint_stack = temp_list.copy()
    else:
        load_in_memory = False 


    count = 0
    for iteration in range(first_iter, final_iter+1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    count +=1
                    viewpoint_index = (count ) % len(video_cams)
                    if (count //(len(video_cams))) % 2 == 0:
                        viewpoint_index = viewpoint_index
                    else:
                        viewpoint_index = len(video_cams) - viewpoint_index - 1
                    # print(viewpoint_index)
                    viewpoint = video_cams[viewpoint_index]
                    custom_cam.time = viewpoint.time
                    # print(custom_cam.time, viewpoint_index, count)
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer, stage=stage, cam_type=scene.dataset_type)["render"]

                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive) :
                    break
            except Exception as e:
                print(e)
                network_gui.conn = None
        
        # Fisher RF Start

        it = iteration
        if stage == "fine":
            # account for the coarse iterations that have already happened
            it += opt.coarse_iterations
        num_views = schema.num_views_to_add(it)
        if num_views > 0:
            try:
                # For sectioned training
                candidate_views_filter = getattr(schema, "candidate_views_filter")[it] if hasattr(schema, "candidate_views_filter") else None
                scene.candidate_views_filter = candidate_views_filter
                
                # Because selection is time consumeing
                selected_views = active_method.nbvs(gaussians, scene, num_views, pipe, background, exit_func=csm.should_exit)
            except RuntimeError as e:
                print(e)
                print("selector exited early")
                # NOTE: we use iteration - 1 because the selector is not done
                save_checkpoint(gaussians, iteration - 1, scene)
                csm.requeue()

            print(f"ITER {it}: selected views: {selected_views}")
            scene.train_idxs.extend(selected_views)
            print(f"ITER {it}: training views after selection: {scene.train_idxs}")

            gaussians.optimizer.zero_grad(set_to_none = True)

            first_iter, _ = load_checkpoint(init_ckpt_path, gaussians, scene, opt, ignore_train_idxs=True)
            base_iter = iteration - 1

        # FisherRF end 

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

            print("Gaussians State Dict: ", gaussians._deformation.state_dict().keys())

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        # dynerf's branch

        # if opt.dataloader and not load_in_memory:
        #     try:
        #         viewpoint_cams = next(loader)
        #     except StopIteration:
        #         print("reset dataloader into random dataloader.")
        #         if not random_loader:
        #             viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=opt.batch_size,shuffle=True,num_workers=32,collate_fn=list)
        #             random_loader = True
        #         loader = iter(viewpoint_stack_loader)

        idx = 0
        viewpoint_cams = []

        while idx < batch_size :    
                
            viewpoint_cam = viewpoint_stack.pop(randint(0,len(viewpoint_stack)-1))
            # if not viewpoint_stack :
            #     viewpoint_stack =  temp_list.copy()
            viewpoint_cams.append(viewpoint_cam)
            idx +=1
        if len(viewpoint_cams) == 0:
            continue

        # print(len(viewpoint_cams))     
        # breakpoint()   
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        images = []
        gt_images = []
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []
        for viewpoint_cam in viewpoint_cams:
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, stage=stage,cam_type=scene.dataset_type)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            images.append(image.unsqueeze(0))
            if scene.dataset_type!="PanopticSports":
                gt_image = viewpoint_cam.original_image.cuda()
            else:
                gt_image  = viewpoint_cam['image'].cuda()
            
            gt_images.append(gt_image.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)
        

        radii = torch.cat(radii_list,0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        image_tensor = torch.cat(images,0)
        gt_image_tensor = torch.cat(gt_images,0)
        # Loss
        # breakpoint()
        Ll1 = l1_loss(image_tensor, gt_image_tensor[:,:3,:,:])

        psnr_ = psnr(image_tensor, gt_image_tensor).mean().double()
        # norm
        

        loss = Ll1
        if stage == "fine" and hyper.time_smoothness_weight != 0:
            # tv_loss = 0
            tv_loss = gaussians.compute_regulation(hyper.time_smoothness_weight, hyper.l1_time_planes, hyper.plane_tv_weight)
            loss += tv_loss
        if opt.lambda_dssim != 0:
            ssim_loss = ssim(image_tensor,gt_image_tensor)
            loss += opt.lambda_dssim * (1.0-ssim_loss)
        # if opt.lambda_lpips !=0:
        #     lpipsloss = lpips_loss(image_tensor,gt_image_tensor,lpips_model)
        #     loss += opt.lambda_lpips * lpipsloss
        
        loss.backward()
        if torch.isnan(loss).any():
            print("loss is nan,end training, reexecv program now.")
            os.execv(sys.executable, [sys.executable] + sys.argv)
        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        for idx in range(0, len(viewspace_point_tensor_list)):
            viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            total_point = gaussians._xyz.shape[0]
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                                          "psnr": f"{psnr_:.{2}f}",
                                          "point":f"{total_point}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            # timer.pause()
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, [pipe, background], stage, scene.dataset_type)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, stage)
            # if dataset.render_process:
            #     if (iteration < 1000 and iteration % 10 == 9) \
            #         or (iteration < 3000 and iteration % 50 == 49) \
            #             or (iteration < 60000 and iteration %  100 == 99) :
                    # breakpoint()
                        # render_training_image(scene, gaussians, [test_cams[iteration%len(test_cams)]], render, pipe, background, stage+"test", iteration,timer.get_elapsed_time(),scene.dataset_type)
                        # render_training_image(scene, gaussians, [train_cams[iteration%len(train_cams)]], render, pipe, background, stage+"train", iteration,timer.get_elapsed_time(),scene.dataset_type)
                        # render_training_image(scene, gaussians, train_cams, render, pipe, background, stage+"train", iteration,timer.get_elapsed_time(),scene.dataset_type)

                    # total_images.append(to8b(temp_image).transpose(1,2,0))
            # timer.start()
            # Densification
            if iteration < opt.densify_until_iter :
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

                if stage == "coarse":
                    opacity_threshold = opt.opacity_threshold_coarse
                    densify_threshold = opt.densify_grad_threshold_coarse
                else:    
                    opacity_threshold = opt.opacity_threshold_fine_init - iteration*(opt.opacity_threshold_fine_init - opt.opacity_threshold_fine_after)/(opt.densify_until_iter)  
                    densify_threshold = opt.densify_grad_threshold_fine_init - iteration*(opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_after)/(opt.densify_until_iter )  
                if  iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0]<360000:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    
                    gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold, 5, 5, scene.model_path, iteration, stage)
                if  iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0 and gaussians.get_xyz.shape[0]>200000:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                    gaussians.prune(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                    
                # if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 :
                if iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0]<360000 and opt.add_point:
                    gaussians.grow(5,5,scene.model_path,iteration,stage)
                    # torch.cuda.empty_cache()
                if iteration % opt.opacity_reset_interval == 0:
                    print("reset opacity")
                    gaussians.reset_opacity()
                    
            

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" +f"_{stage}_" + str(iteration) + ".pth")

        

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, before_selection=False, log_every_image=False):
    if wandb.run is None:
        wandb.init(
            project="Thesis",
            name="run 2",
            resume="allow"  # Allow resuming or starting new
        )
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    # if iteration in testing_iterations or before_selection  (FisherRF)
    if iteration in testing_iterations:
        print(f"Running evaluation for iteration: {iteration}")
        torch.cuda.empty_cache()
        lpips = lpips_func("cuda", net_type='vgg')
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0

                log_images = {}
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and ((idx < 5) or log_every_image):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(idx), image[None], global_step=iteration)
                        log_images[f"render/{idx:03d}"] = wandb.Image(image[None])
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(idx), gt_image[None], global_step=iteration)
                            log_images[f"gt/{idx:03d}"] = wandb.Image(gt_image.cpu()[None])
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                    lpips.to(image.device)
                    lpips_test += lpips(image, gt_image).mean().double()

                if log_every_image:
                    wandb.log(log_images, step=iteration)

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])

                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)
                log_dict = {config['name'] + '/l1_loss': l1_test, config['name'] + '/psnr': psnr_test,
                            config['name'] + '/ssim': ssim_test, config['name'] + '/lpips': lpips_test,}
                wandb.log(log_dict, step=iteration)

        if tb_writer:
            # tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
            wandb.log({'total_points': scene.gaussians.get_xyz.shape[0]}, step=iteration)
        torch.cuda.empty_cache()

import socket
from contextlib import closing

def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")

    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[15_000, 20_000, 25_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    # Flags for view selections
    parser.add_argument("--method", type=str, default="rand")
    parser.add_argument("--schema", type=str, default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reg_lambda", type=float, default=1e-6)
    parser.add_argument("--I_test", action="store_true", help="Use I test to get the selection base")
    parser.add_argument("--I_acq_reg", action="store_true", help="apply reg_lambda to acq H too")
    parser.add_argument("--sh_up_every", type=int, default=5_000, help="increase spherical harmonics every N iterations")
    parser.add_argument("--sh_up_after", type=int, default=-1, help="start to increate active_sh_degree after N iterations")
    parser.add_argument("--min_opacity", type=float, default=0.005, help="min_opacity to prune")
    parser.add_argument("--filter_out_grad", nargs="+", type=str, default=["rotation"])
    parser.add_argument("--log_every_image", action="store_true", help="log every images during traing")
    parser.add_argument("--override_idxs", default=None, type=str, help="speical test idxs on uncertainty evaluation")

    # From 4D Gaussians
    parser.add_argument("--configs", type=str, default = "")

    wandb.init(
        project='Thesis',
        name='run 2',  
    )

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.log_every_image:
        args.test_iterations = []
    if args.iterations not in args.test_iterations:
        args.test_iterations.append(args.iterations)
    
    # if args.start_checkpoint is None:
    #     args.start_checkpoint = args.model_path + "/last.pth"

    print("Args Iterations: ", args.iterations)
    
    # From 4D Gaussians
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)

    print("Optimizing " + args.model_path)

    wandb.init(project='active', resume="allow", id=os.path.split(args.model_path.rstrip('/'))[-1], config=vars(args))

    # Initialize system state (RNG)
    safe_state(args.quiet, seed=args.seed)

    # Start GUI server, configure and run training
    args.port = find_free_port()
    print(f"GUI at: {args.ip}:{args.port}")



    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from,
             args)

    # All done
    print("\nTraining complete.")
