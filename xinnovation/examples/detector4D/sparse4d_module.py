import torch
import torch.nn as nn
from typing import Dict, Optional, Union, Any, List, Tuple

import lightning.pytorch as L
from xinnovation.src.core import LIGHTNING_MODULE, SourceCameraId, TrajParamIndex, EgoStateIndex
from xinnovation.src.components.lightning_module import LightningDetector
from xinnovation.src.utils.math_utils import sample_bbox_edge_points
from xinnovation.src.utils.pose_transform import project_points_to_image
from xinnovation.src.utils.visualize_utils import generate_video_from_dir
from .sparse4d_detector import Sparse4DDetector
from .sparse4d_loss import Sparse4DLossWithDAC
from .sparse4d_dataset import TrainingSample, CAMERA_ID_LIST
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import glob
import shutil
import xinnovation.src.core.training_state as TS
from xinnovation.src.utils.visualize_utils import visualize_query_heatmap
from xinnovation.src.utils.math_utils import generate_bbox2D_from_pixel_cloud, combine_multiple_images

__all__ = ["Sparse4DModule"]

@LIGHTNING_MODULE.register_module()
class Sparse4DModule(LightningDetector):
    
    def __init__(self, detector: Dict, loss: Dict, optimizer: Dict, scheduler: Dict, **kwargs):
        super().__init__(detector=detector, loss=loss, optimizer=optimizer, scheduler=scheduler, **kwargs)

        assert "debug_config" in kwargs, "debug_config is required"
        self.debug_config = kwargs["debug_config"]

        # Initialize validation metrics
        self.val_step_outputs = []
        
        # Initialize predict metrics
        self.predict_step_outputs = []
        
        # for visualization
        if self.debug_config.visualize_validation_results:
            self.register_buffer("bbox_edge_points", sample_bbox_edge_points(20))
            
        # save intermediate results
        TS.intermediate_result_save_dir = f"{self.debug_config.visualize_validation_results_dir}/intermediate_results"
        os.makedirs(TS.intermediate_result_save_dir, exist_ok=True)
        TS.save_intermediate_results = self.debug_config.save_intermediate_results
        
    def forward(self, batch) -> List[Dict]:
        return self.detector(batch)
    
    def _render_trajs_on_imgs(self, 
                      trajs: torch.Tensor, 
                      trajs_prob: torch.Tensor,
                      camera_ids: List[SourceCameraId],
                      calibrations: torch.Tensor,
                      ego_states: torch.Tensor, 
                      imgs_dict: Dict[SourceCameraId, torch.Tensor], 
                      color: torch.Tensor,
                      matched_indices: List[Tuple[np.ndarray, np.ndarray]]=None,
                      refined_trajs: torch.Tensor=None,
                      refined_color: torch.Tensor=None) -> Tuple[Dict[SourceCameraId, torch.Tensor], Dict[SourceCameraId, np.ndarray]]:
        """
        Render trajectories on images.
        Args:
            trajs: [B, N, TrajParamIndex.END_OF_INDEX]
            trajs_prob: [B, N]
            camera_ids: List[SourceCameraId]
            calibrations: torch.Tensor[B, C, CameraParamIndex.END_OF_INDEX]
            ego_states: torch.Tensor[B, T, EgoStateParamIndex.END_OF_INDEX]
            imgs_dict: Dict[SourceCameraId, torch.Tensor[B, T, 3, H, W]]
            color: torch.Tensor[3]
            matched_indices: List[Tuple[np.ndarray, np.ndarray]]
            refined_trajs: torch.Tensor[B, N, TrajParamIndex.END_OF_INDEX] after refinement
        Returns:
            imgs_dict: Dict[SourceCameraId, torch.Tensor[B, T, 3, H, W]]
            concat_imgs: Dict[SourceCameraId, np.ndarray]
        """
        B, T, _ = ego_states.shape
        
        pixels = project_points_to_image(trajs, calibrations, ego_states, self.bbox_edge_points, use_log_dimension=self.detector.use_log_dimension) # pixels: torch.Tensor[B*T, N, C, P, 2]
        _, N, C, P, _ = pixels.shape
        pixels = pixels.view(B, T, N, C, P, 2)
        
        if refined_trajs is not None:
            refined_pixels = project_points_to_image(refined_trajs, calibrations, ego_states, self.bbox_edge_points, use_log_dimension=self.detector.use_log_dimension) # pixels: torch.Tensor[B*T, N, C, P, 2]
            refined_pixels = refined_pixels.view(B, T, N, C, P, 2)
        
        # disable pixels of false positive trajectories
        traj_fp_mask = trajs_prob < self.debug_config.pred_traj_threshold # [B, N]

        if matched_indices is not None:
            traj_fp_mask = torch.ones_like(traj_fp_mask)

            batch_index = [torch.tensor([ib for _ in range(len(match[0]))]) for ib, match in enumerate(matched_indices)]
            batch_index = torch.cat(batch_index, dim=0)
            query_index = torch.cat([torch.tensor(match[1]) for match in matched_indices], dim=0)
            
            traj_fp_mask[batch_index, query_index] = 0

        traj_fp_mask = traj_fp_mask.unsqueeze(1).unsqueeze(3).unsqueeze(4).expand(-1, T, -1, C, P)
        pixels[traj_fp_mask, :] = -1
        
        center_bboxs = generate_bbox2D_from_pixel_cloud(pixels, True) # (B, T, N, C, 5)
        
        if refined_trajs is not None:
            refined_center_bboxs = generate_bbox2D_from_pixel_cloud(refined_pixels, True) # (B, T, N, C, 5)
        
        
        # begin to plot
        concat_imgs: Dict[SourceCameraId, np.ndarray] = {}
        
        # plot pixels on images
        color = color.to(self.device)
        for cam_idx, camera_id in enumerate(camera_ids):
            img_sequence = imgs_dict[camera_id] # [B, T, H, W, 3]
            H, W = img_sequence.shape[2:4]

            tmp_pixel = pixels[:, :, :, cam_idx, ...] # [B, T, N, P, 2]
            tmp_pixel = tmp_pixel.reshape(B, T, N * P, 2) # [B, T, N * P, 2]

            tmp_pixel[:, :, :, 0] *= W
            tmp_pixel[:, :, :, 1] *= H
            tmp_invalid_mask = torch.logical_or(
                torch.logical_or(tmp_pixel[:, :, :, 0] < 0, tmp_pixel[:, :, :, 1] < 0),
                torch.logical_or(tmp_pixel[:, :, :, 0] > W - 1, tmp_pixel[:, :, :, 1] > H - 1)
            )
            
            tmp_pixel[tmp_invalid_mask, :] = -1
            
            h_coords = tmp_pixel[:, :, :, 1].long()
            w_coords = tmp_pixel[:, :, :, 0].long()
            mask = torch.ones_like(img_sequence).to(self.device) # [B, T, H, W, 3]
            
            # 利用PyTorch的高级索引功能，通过向量化操作快速修改像素
            batch_indices = torch.arange(B, device=self.device)[:, None, None].expand(-1, T, N * P)
            time_indices = torch.arange(T, device=self.device)[None, :, None].expand(B, -1, N * P)
            r = (self.debug_config.point_radius + 1) // 2
            r = max(r, 1)
            for i in range(-r, r):
                h_indices = h_coords + i
                h_indices = torch.clamp(h_indices, 0, H - 1)
                for j in range(-r, r):
                    w_indices = w_coords + j
                    w_indices = torch.clamp(w_indices, 0, W - 1)
                    mask[batch_indices, time_indices, h_indices, w_indices, :] = 0
            img_sequence = img_sequence * mask + color * (1 - mask)
            imgs_dict[camera_id] = img_sequence

            # save concat imgs
            cimg = img_sequence.permute(0, 2, 1, 3, 4).contiguous() # [B, H, T, W, 3]
            cimg = cimg.reshape(B * H, T * W, 3) # [B * H, T * W, 3]
            cimg = cimg.cpu().numpy() # [B * H, T * W, 3]
            cimg = cimg.astype(np.uint8)
            
            # BBox 2D and its center
            tmp_bbox = center_bboxs[..., cam_idx, :] # [B, T, N, 5]
            tmp_bbox = tmp_bbox.reshape(B, T, N, 5) # [B, T, N, 5]
            
            if refined_trajs is not None:
                refined_tmp_bbox = refined_center_bboxs[..., cam_idx, :] # [B, T, N, 5]
                refined_tmp_bbox = refined_tmp_bbox.reshape(B, T, N, 5) # [B, T, N, 5]
                
                
            def draw_info_on_img(bbox_, color_, cimg_, H_, W_, id_, plot_bbox_):
                cx, cy, w, h, valid = bbox_
                if not valid:
                    return
                cx, cy, w, h = cx.item(), cy.item(), w.item(), h.item()
                cx *= W_
                cy *= H_
                w *= W_
                h *= H_
                x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # draw bbox
                if plot_bbox_:
                    cv2.rectangle(cimg_, (x1, y1), (x2, y2), color_.cpu().numpy().tolist(), 1)
                # draw traj id
                text = f"{id_}"
                cv2.putText(cimg_, text, (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_.cpu().numpy().tolist(), 1)
                
                
            # Visualize the text info on image, such as the trajectory id
            if matched_indices is not None:
                for ib, match in enumerate(matched_indices):
                    traj_inds = match[1]
                    matched_bboxs = tmp_bbox[ib, :, traj_inds]  # [T, K, 5]
                    for it in range(T):
                        for ik in range(len(traj_inds)):
                            bbox = matched_bboxs[it][ik]
                            # draw coarse matched bbox info
                            draw_info_on_img(bbox, color, cimg, H, W, traj_inds[ik], False)
                            if refined_trajs is not None:
                                # draw refined matched bbox info
                                refined_bbox = refined_tmp_bbox[ib, it, traj_inds[ik]]
                                draw_info_on_img(refined_bbox, refined_color, cimg, H, W, traj_inds[ik], True)
            if T > 1:
                # we plot the timestamp info when T > 1, sequential model is activated
                for ib in range(B):
                    for it in range(T):
                        # debug_message = f"{camera_id.name}: {ego_states[ib, it, EgoStateIndex.STEADY_TIMESTAMP]:.2f}"
                        debug_message = f"{ego_states[ib, it, EgoStateIndex.STEADY_TIMESTAMP]:.2f}"
                        cv_color = color.cpu().numpy().tolist()
                        # Extract the tile and create a contiguous copy
                        tile = cimg[ib * H : (ib + 1) * H, it * W : (it + 1) * W].copy()

                        # Draw text on the contiguous tile
                        cv2.putText(tile, debug_message, (W // 2, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, cv_color, 2)

                        # Update the original image with the modified tile
                        cimg[ib * H : (ib + 1) * H, it * W : (it + 1) * W] = tile
                        # cv2.putText(cimg[ib * H: (ib + 1) * H, it * W: (it + 1) * W], debug_message, (W // 2, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, cv_color, 2)
            cimg = cimg.astype(np.uint8)
            concat_imgs[camera_id] = cimg
            
            # NOTE: img_sequence[batch_indices, time_indices, tmp_pixel[:, :, :, 1], tmp_pixel[:, :, :, 0], :] = color
            # will report error: img_sequence[batch_indices, time_indices, tmp_pixel[:, :, :, 1], tmp_pixel[:, :, :, 0], :] = color
            # RuntimeError: linearIndex.numel()*sliceSize*nElemBefore == expandedValue.numel() INTERNAL ASSERT FAILED at 
            # "/pytorch/aten/src/ATen/native/cuda/Indexing.cu":548, please report a bug to PyTorch. number of flattened indices 
            # did not match number of elements in the value tensor: 15360 vs 9

            debug = False
            if debug:
                cv2.imwrite('img2.png', cimg)
                cv2.imshow('img', cimg)
                cv2.waitKey(0)
                exit(0)
    
        return imgs_dict, concat_imgs
    
    def on_train_batch_start(self, batch, batch_idx):
        """On train batch start."""
        pass

    def on_train_epoch_start(self):
        """On train epoch start."""
        self.criterion.update_epoch(self.current_epoch)
            
    def on_train_start(self):
        """On train start."""
        print("On train start")
        # 确保重置匹配历史记录
        self.criterion.reset_matching_history()

        # create log dir
        os.makedirs(self.debug_config.log_dir, exist_ok=True)

        # save config file
        config_path = os.path.join(self.debug_config.log_dir, "train_config.json")
        with open(config_path, 'w') as f:
            json.dump(self.hparams, f, indent=2)

    def _visualize_matching_results(self):
        """Save the matching history."""
        print("Visualizing matching history...")
        # 获取匹配历史记录
        matching_history = self.criterion.matching_history 
        # layer_idx -> epoch_idx -> batch_idx -> Tuple[np.ndarray, np.ndarray]
        
        if not matching_history:
            print("No matching history to visualize.")
            return
        
        vis_dir = os.path.join(self.debug_config.visualize_validation_results_dir, "query_assignment_changes")
        os.makedirs(vis_dir, exist_ok=True)
        
        # 获取所有层索引
        layer_indices = sorted(matching_history.keys())
        print(f"Found data for {len(layer_indices)} decoder layers")
        
        # 对每个层创建折线图可视化
        for layer_idx in layer_indices:
            # 获取该层的所有匹配记录
            layer_matches = matching_history[layer_idx]
            print(f"Layer {layer_idx}: Found {len(layer_matches)} epochs of match records")
            
            # 收集每个batch的匹配信息
            batch_gt_to_query = {}  # batch_idx -> List[np.ndarray], pred indices of each gt
            
            # Process matches for each epoch
            for _, epoch_matches in enumerate(layer_matches): # List[List[Tuple[np.ndarray, np.ndarray]]]
                for batch_idx, batch_matches in enumerate(epoch_matches): # List[Tuple[np.ndarray, np.ndarray]]
                    if batch_idx not in batch_gt_to_query:
                        batch_gt_to_query[batch_idx] = []
                    batch_gt_to_query[batch_idx].append(batch_matches[1])
            epochs = list(range(len(layer_matches)))
            # length = [p.size for p in pred_indices]
            for batch_idx, pred_indices in batch_gt_to_query.items():
                ys = np.stack(pred_indices, axis=0).T # shape: [num_gt, num_epoch]
                plt.figure(figsize=(14, 8))
                
                # Create color map for different GT indices
                colors = plt.cm.rainbow(np.linspace(0, 1, ys.shape[0]))
                
                for i in range(ys.shape[0]):
                    plt.plot(epochs, ys[i], 'o-', linewidth=1.5,
                            label=f'GT {i}',
                            color=colors[i],
                            markersize=4)
                plt.grid(True)
                plt.xlabel('Training Epoch', fontsize=12)
                plt.ylabel('Matched Query Index', fontsize=12)
                plt.title(f'Query Assignment Changes Over Training\n(Layer {layer_idx}, Batch {batch_idx})', 
                          fontsize=14, fontweight='bold')
                plt.legend()
                plt.tight_layout()
                save_path = os.path.join(vis_dir, f'layer_{layer_idx}_batch_{batch_idx}_matching.png')
                plt.savefig(save_path, dpi=300)
                plt.close()
                print(f"Saved matching visualization for Layer {layer_idx}, Batch {batch_idx}")
        
        print(f"Matching visualization completed. Results saved to {vis_dir}")
    
    def _visualize_trajs_on_bev(self):
        """Generate the video of the predicted trajectories."""
        if not self.debug_config.visualize_validation_results:
            return
        print("Generating the video of the predicted trajectories...")
        imgs_dir1 = f'{self.criterion.val_debug_dir}/matched_trajs_on_bev'
        imgs_dir2 = f'{self.criterion.val_debug_dir}/refined_trajs_on_bev'
               
        generate_video_from_dir(imgs_dir1, os.path.join(self.debug_config.visualize_validation_results_dir, "matched_trajs_on_bev.mp4"), 10)
        generate_video_from_dir(imgs_dir2, os.path.join(self.debug_config.visualize_validation_results_dir, "refined_trajs_on_bev.mp4"), 10)
        print("Finished generating BEV matching results video")
        
    def _visualize_query_heatmap(self):
        """Visualize the query heatmap."""
        if not self.debug_config.visualize_validation_results:
            return
        print("Generating the query heatmap...")
        query_heatmap_dir = os.path.join(TS.intermediate_result_save_dir, "query_heatmap")
        generate_video_from_dir(query_heatmap_dir, os.path.join(self.debug_config.visualize_validation_results_dir, "query_heatmap.mp4"), 10)

    def _generate_validation_trajs_video(self):
        """Generate the video of the predicted trajectories."""
        if not self.debug_config.visualize_validation_results:
            return
        print("Generating the video of the predicted trajectories...")
        trajs_dir = os.path.join(self.debug_config.visualize_validation_results_dir, "pred_trajs")
        if not os.path.exists(trajs_dir):
            print("No predicted trajectories to visualize")
            return
        
        img_files = glob.glob(os.path.join(trajs_dir, "*.png"))
        
        for camera_id in self.debug_config.visualize_camera_list:
            camera_name = camera_id.name
            camera_img_files = [file for file in img_files if camera_name in file]
            camera_img_files.sort()
            
            # Save frames to a temporary directory
            temp_dir = os.path.join(trajs_dir, f"temp_{camera_name}")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Copy and rename frames sequentially for ffmpeg
            for i, img_file in enumerate(camera_img_files):
                img = cv2.imread(img_file)
                if img is not None:
                    frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
                    # plot "Epoch xxx" on the top middle of the image
                    cv2.putText(img, f"Epoch {self.current_epoch}", (img.shape[1] // 2, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imwrite(frame_path, img)
            
            video_path = os.path.join(self.debug_config.visualize_validation_results_dir, f"pred_trajs_{camera_name}.mp4")
            
            # Use ffmpeg to create the video
            
            ffmpeg_cmd = f"ffmpeg -y -framerate 10 -i {temp_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p {video_path}"
            # ffmpeg_cmd = f"ffmpeg -y -framerate 10 -i {temp_dir}/frame_%04d.png -c:v libopenh264 -pix_fmt yuv420p {video_path}"
            # ffmpeg_cmd = f"ffmpeg -y -framerate 10 -i {temp_dir}/frame_%04d.png -c:v mpeg4 -q:v 1 -pix_fmt yuv420p {video_path}"
            os.system(ffmpeg_cmd)
            
            # Clean up temporary directory
            shutil.rmtree(temp_dir)
        shutil.rmtree(trajs_dir)
        print("Predicted trajectories video generation completed")
    
    def _generate_matched_trajs_video(self):
        """Generate the video of the matched trajectories."""
        if not self.debug_config.visualize_validation_results:
            return
        print("Generating the video of the matched trajectories...")
        trajs_dir = os.path.join(self.debug_config.visualize_validation_results_dir, "matched_trajs")
        video_save_dir = os.path.join(self.debug_config.visualize_validation_results_dir, "matched_trajs_detail_video")
        os.makedirs(video_save_dir, exist_ok=True)
        if not os.path.exists(trajs_dir):
            print("No matched trajectories to visualize")
            return
        
        # Get all image files
        img_files = glob.glob(os.path.join(trajs_dir, "*.png"))
        if not img_files:
            print("No image files found")
            return
            
        # Helper function to extract epoch and layer from filename
        def get_epoch_layer(filename):
            # Format: "FRONT_CENTER_CAMERA_00000_0.png"
            base = os.path.basename(filename)
            parts = base.split('_')
            # The last two parts are epoch and layer
            epoch_str = parts[-2]  # "00000"
            layer_str = parts[-1].split('.')[0]  # "0" (remove .png)
            return int(epoch_str), int(layer_str)
            
        # Group files by camera and layer
        camera_layer_files = {}  # (camera_name, layer_idx) -> files
        camera_files = {}  # camera_name -> files (all layers)
        
        for camera_id in self.debug_config.visualize_camera_list:
            camera_name = camera_id.name
            camera_files[camera_name] = []
            
            # Get all files for this camera
            camera_img_files = [f for f in img_files if camera_name in f]
            if not camera_img_files:
                continue
                
            # Group by layer
            for layer_idx in range(6):
                layer_files = [f for f in camera_img_files if f"_{layer_idx}.png" in f]
                if layer_files:
                    # Sort layer files by epoch
                    layer_files.sort(key=lambda x: get_epoch_layer(x)[0])  # sort by epoch number
                    key = (camera_name, layer_idx)
                    camera_layer_files[key] = layer_files
                    
            # Sort all camera files by epoch and layer
            camera_files[camera_name] = sorted(camera_img_files, key=lambda x: get_epoch_layer(x))
        
        # 1. Generate per-layer videos
        print("Generating per-layer videos...")
        for (camera_name, layer_idx), files in camera_layer_files.items():
            print(f"Processing camera {camera_name}, layer {layer_idx}")
            
            # Create temporary directory for frames
            temp_dir = os.path.join(trajs_dir, f"temp_{camera_name}_layer_{layer_idx}")
            os.makedirs(temp_dir, exist_ok=True)
            
            try:
                # Copy and rename frames sequentially
                for i, img_file in enumerate(files):
                    frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
                    shutil.copy2(img_file, frame_path)
                
                # Generate video using ffmpeg
                video_name = f"matched_trajs_{camera_name}_layer_{layer_idx}.mp4"
                video_path = os.path.join(video_save_dir, video_name)
                
                ffmpeg_cmd = f"ffmpeg -y -framerate 10 -i {temp_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p {video_path}"
                os.system(ffmpeg_cmd)
                print(f"Generated layer video: {video_path}")
                
            finally:
                # Clean up temporary directory
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
        
        # 2. Generate combined videos (all layers per camera)
        print("\nGenerating combined-layer videos...")
        for camera_name, files in camera_files.items():
            if not files:
                continue
                
            print(f"Processing camera {camera_name} (all layers)")
            
            # Create temporary directory for frames
            temp_dir = os.path.join(trajs_dir, f"temp_{camera_name}_combined")
            os.makedirs(temp_dir, exist_ok=True)
            
            try:
                # Copy and rename frames sequentially
                for i, img_file in enumerate(files):
                    frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
                    shutil.copy2(img_file, frame_path)
                
                # Generate video using ffmpeg
                video_name = f"matched_trajs_{camera_name}_combined.mp4"
                video_path = os.path.join(video_save_dir, video_name)
                
                ffmpeg_cmd = f"ffmpeg -y -framerate 10 -i {temp_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p {video_path}"
                os.system(ffmpeg_cmd)
                print(f"Generated combined video: {video_path}")
                
            finally:
                # Clean up temporary directory
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
        
        # 3. Generate combined videos (combine all image to one image)
        print("\nGenerating combined-camera videos...")
        
        # Create temporary directory for frames
        temp_dir = os.path.join(trajs_dir, f"temp_all_cameras_combined")
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            camera_ids = [cam.name for cam in self.debug_config.visualize_camera_list]
            file_nb = len(camera_files['FRONT_LEFT_CAMERA'])
            for i in range(file_nb):
                tmp_img_list = []
                for camera in camera_ids:
                    tmp_img_list.append(cv2.imread(camera_files[camera][i]))
                combined_img = combine_multiple_images(tmp_img_list)
                frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
                cv2.imwrite(frame_path, combined_img)
                
            video_name = f"matched_trajs_all_cameras_combined.mp4"
            video_path = os.path.join(self.debug_config.visualize_validation_results_dir, video_name)
            
            ffmpeg_cmd = f"ffmpeg -y -framerate 10 -i {temp_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p {video_path}"
            os.system(ffmpeg_cmd)
            print(f"Generated combined video: {video_path}")
            
        finally:
            # Clean up temporary directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                
        # shutil.rmtree(trajs_dir)
        print("Matched trajectories video generation completed")

    def on_train_end(self):
        """On train end, visualize the matching history."""
        if self.global_rank == 0:
            # self._generate_validation_trajs_video()
            self._generate_matched_trajs_video()
            self._visualize_matching_results()
            self._visualize_trajs_on_bev()
            self._visualize_query_heatmap()
        
    def on_validation_end(self):
        """On validation end, visualize the matching history."""
        pass
        
    def visualize_init_trajs(self, batch: Dict):
        """Visualize initial trajectories."""
        # 1. read images
        imgs_dict = TrainingSample.read_seqeuntial_images_to_tensor(batch['image_paths'], self.device)
        # 2. render init trajs
        init_trajs = self.detector.get_init_trajs(batch['ego_states'].shape[0])
        init_trajs[..., TrajParamIndex.HAS_OBJECT] = 1.0
        imgs_dict, concat_imgs = self._render_trajs_on_imgs(init_trajs, 
                                                init_trajs[..., TrajParamIndex.HAS_OBJECT],
                                                batch['camera_ids'],
                                                batch['calibrations'],
                                                batch['ego_states'], 
                                                imgs_dict,
                                                color=torch.tensor(self.debug_config.init_color))
        
        # 3. save images
        save_dir = os.path.join(self.debug_config.visualize_validation_results_dir, "init_trajs")
        os.makedirs(save_dir, exist_ok=True)
        for camera_id in concat_imgs.keys():
            img_name = f'Init_trajs_{camera_id.name}.png'
            img = cv2.cvtColor(concat_imgs[camera_id], cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(save_dir, img_name), img)

    def save_validation_intermediate_results(self, batch: Dict, batch_idx: int, trajs_list: List[torch.Tensor]):
        """
        Save intermediate results for visualization, only save the first batch for training and validation.
        Args:
            batch: Dict
            batch_idx: int
            trajs_list: List[torch.Tensor], trajs of all decoder layers
        """
        if self.global_rank != 0:
            return
        if not (self.debug_config.visualize_validation_results and batch_idx == 0):
            return
        
        epoch_str = f"{self.current_epoch:05d}"
        # 1. read images
        imgs_dict = TrainingSample.read_seqeuntial_images_to_tensor(batch['image_paths'], self.device)

        # 2. render gt trajs
        if self.debug_config.render_gt_trajs:
            imgs_dict, concat_imgs = self._render_trajs_on_imgs(batch['trajs'], 
                                                    batch['trajs'][..., TrajParamIndex.HAS_OBJECT],
                                                    batch['camera_ids'],
                                                    batch['calibrations'],
                                                    batch['ego_states'], 
                                                    imgs_dict,
                                                    color=torch.tensor(self.debug_config.gt_color))
            
        # 3. render matched trajs
        if self.debug_config.render_matched_trajs:
            # only train mode stores the matching history
            save_dir = os.path.join(self.debug_config.visualize_validation_results_dir, "matched_trajs")
            os.makedirs(save_dir, exist_ok=True)

            num_decoder_layers = len(trajs_list)
            # Define colors for different layers using rainbow colormap
            layer_colors = plt.cm.rainbow(np.linspace(0, 1, num_decoder_layers))  # 6 layers
            layer_colors = (layer_colors[:, :3] * 255).astype(np.int32)  # Convert to RGB 0-255 format

            for layer_idx in range(num_decoder_layers):
                imgs_copy = imgs_dict.copy()
                matched_indices = self.criterion.get_latest_matching_indices(layer_idx)
                if matched_indices is None or layer_idx == 0:
                    continue
                coarse_trajs, refined_trajs = trajs_list[layer_idx - 1], trajs_list[layer_idx]
                concat_imgs = self._render_trajs_on_imgs(coarse_trajs, 
                                                    coarse_trajs[..., TrajParamIndex.HAS_OBJECT].sigmoid(),
                                                    batch['camera_ids'],
                                                    batch['calibrations'],
                                                    batch['ego_states'], 
                                                    imgs_copy,
                                                    color=torch.tensor(layer_colors[layer_idx - 1].tolist(), dtype=torch.float32),
                                                    matched_indices=matched_indices,
                                                    refined_trajs=refined_trajs,
                                                    refined_color=torch.tensor(layer_colors[layer_idx].tolist(), dtype=torch.float32))[1]
                for camera_id in concat_imgs.keys():
                    if camera_id not in self.debug_config.visualize_camera_list:
                        continue
                    img_name = f'{camera_id.name}_{epoch_str}_{layer_idx}.png'
                    img = concat_imgs[camera_id].copy()
                    # Add color indicator in the corner
                    color_box = np.zeros((30, 30, 3), dtype=np.uint8)
                    color_box[:] = layer_colors[layer_idx]
                    img[10:40, 10:40] = color_box
                    cv2.putText(img, f"Epoch {self.current_epoch}, Layer {layer_idx}", (img.shape[1] // 2 - 100, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, layer_colors[layer_idx].tolist(), 2)
                    # Convert RGB to BGR before saving
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(save_dir, img_name), img)
                    
                    concat_imgs[camera_id] = img


        # 4. visualize pred trajs of last layer
        if self.debug_config.render_pred_trajs:
            trajs = trajs_list[-1]
            imgs_dict, concat_imgs = self._render_trajs_on_imgs(trajs, 
                                                    trajs[..., TrajParamIndex.HAS_OBJECT].sigmoid(),
                                                    batch['camera_ids'],
                                                    batch['calibrations'],
                                                    batch['ego_states'], 
                                                    imgs_dict,
                                                    color=torch.tensor(self.debug_config.pred_color))
        
        # 5. save images
        save_dir = os.path.join(self.debug_config.visualize_validation_results_dir, "pred_trajs")
        os.makedirs(save_dir, exist_ok=True)
        for camera_id in concat_imgs.keys():
            if camera_id not in self.debug_config.visualize_camera_list:
                continue
            img_name = f'epoch_{epoch_str}_{camera_id.name}.png'
            img = cv2.cvtColor(concat_imgs[camera_id], cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(save_dir, img_name), img)
    
    def update_global_training_state(self, epoch, step, on_val=False):
        TS.current_epoch = epoch
        TS.global_step = step
        TS.on_val = on_val
        
    def training_step(self, batch: Dict, batch_idx: int) -> Dict:
        self.update_global_training_state(self.current_epoch, self.global_step, False)
        """Training step."""
        # Forward pass
        outputs, _, c_outputs, _ = self(batch)
        loss_dict = self.criterion(batch['trajs'], outputs, c_outputs)
        
        # if self.debug_config.render_init_trajs:
        #     self.visualize_init_trajs(batch)
        #     exit()
        
        # # Log losses
        for name, value in loss_dict.items():
            # if name not in ["loss", "standard_decoder_loss", "cross_attention_decoder_loss"]:
            #     continue
            self.log(f"train/{name}", value, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch['trajs'].shape[0], sync_dist=True)
            
        return loss_dict
    
    def validation_step(self, batch: Dict, batch_idx: int) -> Dict:
        self.update_global_training_state(self.current_epoch, self.global_step, True)
        """Validation step."""
        # Forward pass
        outputs, _, c_outputs, _ = self(batch)
        
        if self.debug_config.render_init_trajs:
            self.visualize_init_trajs(batch)
            # exit()
        
        # Compute loss
        loss_dict = self.criterion(batch['trajs'], outputs, c_outputs, batch_idx, "val") # don't save matching history for validation

        self.save_validation_intermediate_results(batch, batch_idx, outputs)
        
        # Store outputs for epoch end
        self.val_step_outputs.append({
            'loss_dict': loss_dict,
            'outputs': outputs[-1],  # Only keep final predictions
            'targets': batch
        })
        
        # Log losses
        for name, value in loss_dict.items():
            if name not in ["loss", "standard_decoder_loss", "cross_attention_decoder_loss"]:
                continue
            self.log(f"val/{name}", value, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch['trajs'].shape[0], sync_dist=True)
        
        return loss_dict
    
    def on_validation_epoch_end(self):
        """Compute validation metrics at epoch end."""
        if not self.val_step_outputs:
            print("No validation outputs to process")
            return
        
        print(f"Processing {len(self.val_step_outputs)} validation outputs")
        
        # Aggregate predictions and targets
        all_preds = []
        all_targets = []
        
        for i, output in enumerate(self.val_step_outputs):
            print(f"Processing validation output {i+1}/{len(self.val_step_outputs)}")
            
            # 这是一个张量，不是字典
            pred_trajs = output['outputs']
            gt_trajs = output['targets']['trajs']
            
            print(f"  Predictions shape: {pred_trajs.shape}")
            print(f"  Targets shape: {gt_trajs.shape}")
            
            # Filter valid predictions and targets
            valid_mask_preds = torch.sigmoid(pred_trajs[..., TrajParamIndex.HAS_OBJECT]) > self.debug_config.pred_traj_threshold  # HAS_OBJECT flag
            valid_mask_targets = gt_trajs[..., TrajParamIndex.HAS_OBJECT] > self.debug_config.pred_traj_threshold
            
            print(f"  Valid predictions: {valid_mask_preds.sum().item()}")
            print(f"  Valid targets: {valid_mask_targets.sum().item()}")
            
            valid_preds = pred_trajs[valid_mask_preds]
            valid_targets = gt_trajs[valid_mask_targets]
            
            all_preds.extend(valid_preds)
            all_targets.extend(valid_targets)
        
        print(f"Total valid predictions: {len(all_preds)}")
        print(f"Total valid targets: {len(all_targets)}")
        
        # Compute metrics
        metrics = self._compute_metrics(all_preds, all_targets)
        
        # Log metrics
        for name, value in metrics.items():
            if name not in ["loss", "standard_decoder_loss", "cross_attention_decoder_loss"]:
                continue
            self.log(f"val/{name}", value, on_epoch=True, sync_dist=True)
        
        # Clear outputs
        self.val_step_outputs.clear()
    
    def test_step(self, batch: Dict, batch_idx: int) -> Dict:
        """Test step."""
        return self.validation_step(batch, batch_idx)
    
    def on_predict_start(self):
        """On predict start."""
        print("On predict start")
        # save config file
        os.makedirs(self.debug_config.predict_dir, exist_ok=True)
        config_path = os.path.join(self.debug_config.predict_dir, "predict_config.json")
        with open(config_path, 'w') as f:
            json.dump(self.hparams, f, indent=2)
    
    def _save_pred_trajs_images(self, batch: Dict, batch_idx: int, outputs: List[torch.Tensor]):
        """Save predicted trajectories images."""
        step_str = f"{batch_idx:05d}"
        # 1. read images
        imgs_dict = TrainingSample.read_seqeuntial_images_to_tensor(batch['image_paths'], self.device)

        # 2. render pred trajs
        num_decoder_layers = len(outputs)
        layer_colors = plt.cm.rainbow(np.linspace(0, 1, num_decoder_layers))  # 6 layers
        layer_colors = (layer_colors[:, :3] * 255).astype(np.int32)  # Convert to RGB 0-255 format

        for layer_idx in range(num_decoder_layers):
            imgs_copy = imgs_dict.copy()
            trajs = outputs[layer_idx]
            imgs_copy, concat_imgs = self._render_trajs_on_imgs(trajs, 
                                                    trajs[..., TrajParamIndex.HAS_OBJECT].sigmoid(),
                                                    batch['camera_ids'],
                                                    batch['calibrations'],
                                                    batch['ego_states'], 
                                                    imgs_copy,
                                                    color=torch.tensor(layer_colors[layer_idx].tolist(), dtype=torch.float32))
            
            # 3. save images
            save_dir = os.path.join(self.debug_config.predict_dir, "pred_trajs")
            os.makedirs(save_dir, exist_ok=True)
            height, width = 540, 960 * self.detector.sequence_length()
            img_list = []
            for camera_id in concat_imgs.keys():
                # Add color indicator in the corner
                img = concat_imgs[camera_id]
                color_box = np.zeros((30, 30, 3), dtype=np.uint8)
                color_box[:] = layer_colors[layer_idx]
                img[10:40, 10:40] = color_box
                cv2.putText(img, f"Layer {layer_idx}", (img.shape[1] // 2 - 100, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, layer_colors[layer_idx].tolist(), 2)
                img = cv2.resize(img, (width, height))
                img_list.append(img)
            img_list = np.vstack(img_list)
            img_list = cv2.cvtColor(img_list, cv2.COLOR_RGB2BGR)
            img_name = f'predict_{step_str}_layer_{layer_idx}.png'
            cv2.imwrite(os.path.join(save_dir, img_name), img_list)
    
    def _generate_predict_trajs_video(self):
        """Generate predicted trajectories video."""
        save_dir = os.path.join(self.debug_config.predict_dir, "pred_trajs")
        if not os.path.exists(save_dir):
            print("No prediction images found")
            return
            
        # Create temporary directory for renamed frames
        temp_dir = os.path.join(save_dir, "temp_frames")
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # Get all prediction images and sort them
            img_files = [f for f in os.listdir(save_dir) if f.endswith('.png') and not f.startswith('frame_')]
            img_files.sort()
            
            # 1. Generate combined video (all layers)
            print("Generating combined video of all layers...")
            for i, img_file in enumerate(img_files):
                src_path = os.path.join(save_dir, img_file)
                dst_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
                shutil.copy2(src_path, dst_path)
            
            video_name = os.path.join(self.debug_config.predict_dir, "predict_trajs_combined.mp4")
            ffmpeg_cmd = f"ffmpeg -y -framerate 10 -i {temp_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p {video_name}"
            os.system(ffmpeg_cmd)
            print(f"Generated combined video: {video_name}")
            
            # Clean up temp directory for next use
            for f in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, f))
            
            # 2. Generate individual videos for each layer
            num_layers = len(self.detector.decoder_layers)
            for layer_idx in range(num_layers):
                print(f"Generating video for layer {layer_idx}...")
                layer_files = [f for f in img_files if f"layer_{layer_idx}" in f]
                layer_files.sort()
                
                for i, img_file in enumerate(layer_files):
                    src_path = os.path.join(save_dir, img_file)
                    dst_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
                    shutil.copy2(src_path, dst_path)
                
                layer_video_name = os.path.join(self.debug_config.predict_dir, f"predict_trajs_layer_{layer_idx}.mp4")
                ffmpeg_cmd = f"ffmpeg -y -framerate 10 -i {temp_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p {layer_video_name}"
                os.system(ffmpeg_cmd)
                print(f"Generated video for layer {layer_idx}: {layer_video_name}")
                
                # Clean up temp directory for next layer
                for f in os.listdir(temp_dir):
                    os.remove(os.path.join(temp_dir, f))
            
        finally:
            # Clean up temporary directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    def predict_step(self, batch: Dict, batch_idx: int, dataloader_idx: int = 0) -> Dict:
        """Prediction step."""
        outputs, _, c_outputs, _ = self(batch)
        self.predict_step_outputs.append(outputs)

        # visualize pred trajs
        if self.debug_config.render_pred_trajs:
            self._save_pred_trajs_images(batch, batch_idx, outputs)
        return outputs
    
    def on_predict_epoch_end(self):
        """On predict epoch end."""
        print("On predict epoch end")
        
    def on_predict_end(self):
        """On predict end."""
        print("On predict end")
        self._generate_predict_trajs_video()
        
    def _compute_metrics(self, predictions: List[torch.Tensor], targets: List[torch.Tensor]) -> Dict:
        """Compute evaluation metrics."""
        metrics = {}
        
        # TODO: Implement detailed metrics computation
        # - Position error
        # - Velocity error
        # - Classification accuracy
        # - Detection metrics (precision, recall, F1)
        
        return metrics 
        