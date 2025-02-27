/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{	
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// if (idx == 0) { // Print only for the first index
	// 	printf("TRYING TO PRINT RESULT\n");
	// 	printf("Result for idx=0: (%f, %f, %f)\n", result.x, result.y, result.z);
	// }


	// if (clamped == nullptr) {
	// 	printf("Error: 'clamped' pointer is null.\n");
	// 	return glm::vec3(0.0f);
	// }

	// Check the bounds for clamped
	// int clamped_size = /* Total size of the clamped array */;
	// if (3 * idx + 2 >= clamped_size) {
	// 	printf("Error: 'clamped' access out of bounds. idx=%d, clamped_size=%d, required=%d\n",
	// 		idx, clamped_size, 3 * idx + 2);
	// 	return glm::vec3(0.0f);
	// }


	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result[0] < 0);
	clamped[3 * idx + 1] = (result[1] < 0);
	clamped[3 * idx + 2] = (result[2] < 0);


	// bool local_clamped[3];
	// local_clamped[0] = (result[0] < 0);
	// local_clamped[1] = (result[1] < 0);
	// local_clamped[2] = (result[2] < 0);

	// if (idx < 0 || idx >= 2000) {
	// 	printf("Error: Invalid idx value: %d\n", idx);
	// 	return;
	// }

	// Note: The results < 0 seem to be working fine -> has to be the clamped is going out of dimension

	// auto res = result[0] < 0;
	// auto res_1 = result[1] < 0;
	// auto res_2 = result[2] < 0;


	// auto clamp = clamped[0];
	// clamped[3 * 0 + 0] = (result[0] < 0);
	// clamped[3 * 0 + 1] = (result[1] < 0);
	// clamped[3 * 0 + 2] = (result[2] < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx == 0) {
        printf("Thread 0: Entering computeCov2D\n");
    }

    // Check for nullptr before accessing memory
    if (!cov3D || !viewmatrix) {
        if (idx == 0) {
            printf("ERROR: NULL pointer in computeCov2D! cov3D=%p, viewmatrix=%p\n", cov3D, viewmatrix);
        }
        return make_float3(0, 0, 0);
    }

	if (idx == 0) {
		if (isnan(mean.x) || isnan(mean.y) || isnan(mean.z) ||
			isinf(mean.x) || isinf(mean.y) || isinf(mean.z)) {
			printf("ERROR: mean contains NaN or Inf! (%f, %f, %f)\n", mean.x, mean.y, mean.z);
		}
	}

    // Safe transformation
    float3 t = transformPoint4x3(mean, viewmatrix);

    // Check for NaN values
    // if (isnan(t.x) || isnan(t.y) || isnan(t.z) || isinf(t.x) || isinf(t.y) || isinf(t.z)) {
    //     if (idx == 0) {
    //         printf("ERROR: NaN or Inf detected in transformed point (%f, %f, %f)\n", t.x, t.y, t.z);
    //     }
    //     return make_float3(0, 0, 0);
    // }

    const float limx = 1.3f * tan_fovx;
    const float limy = 1.3f * tan_fovy;
    const float txtz = t.x / t.z;
    const float tytz = t.y / t.z;
    t.x = min(limx, max(-limx, txtz)) * t.z;
    t.y = min(limy, max(-limy, tytz)) * t.z;

    // Check if cov3D is accessing out of bounds
    // if (idx == 0) {
    //     printf("tx: %f\n", t.x);
    // }

    float c0 = cov3D[0];
    float c1 = cov3D[1];
    float c2 = cov3D[2];
    float c3 = cov3D[3];
    float c4 = cov3D[4];
    float c5 = cov3D[5];

    // Check for NaNs or invalid values
    if (isnan(c0) || isnan(c1) || isnan(c2) || isnan(c3) || isnan(c4) || isnan(c5)) {
        if (idx == 0) {
            printf("ERROR: NaN or Inf in cov3D! (%f, %f, %f, %f, %f, %f)\n", c0, c1, c2, c3, c4, c5);
        }
        return make_float3(0, 0, 0);
    }

    // glm matrix definitions
    glm::mat3 J = glm::mat3(
        focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
        0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
        0, 0, 0);

    glm::mat3 W = glm::mat3(
        viewmatrix[0], viewmatrix[4], viewmatrix[8],
        viewmatrix[1], viewmatrix[5], viewmatrix[9],
        viewmatrix[2], viewmatrix[6], viewmatrix[10]);

    glm::mat3 T = W * J;

    glm::mat3 Vrk = glm::mat3(
        c0, c1, c2,
        c1, c3, c4,
        c2, c4, c5);

    glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

    // Apply low-pass filter: every Gaussian should be at least one pixel wide/high.
    cov[0][0] += 0.3f;
    cov[1][1] += 0.3f;

    // if (idx == 0) {
    //     printf("Thread 0: Computed cov values: (%f, %f, %f)\n", cov[0][0], cov[0][1], cov[1][1]);
    // }

    return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}


// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{	
	printf("IN COMPUTER COV 3D\n");
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)

{	
	// printf("Kernel launched! BlockIdx: %d, ThreadIdx: %d, GlobalIdx: %d\n",
	// 	blockIdx.x, threadIdx.x, threadIdx.x + blockIdx.x * blockDim.x);
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx == 0){
		printf("In preprocess cuda +++++++++++_+_+_+_+_+__+_+_+_+__\n");
	}
	if (idx >= P){
		return;
	}

	if (orig_points == nullptr) {
		if (idx == 0) {
			printf("ERROR: orig_points is NULL in the kernel!\n");
		}
		return;
	}


	// if (idx == 0){
	// 	printf("AFTER RETURN\n");
	// }

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// int temp_radii = 0;
	// cudaMemcpy(radii, &temp_radii, sizeof(int), cudaMemcpyHostToDevice);


	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}



	// Compute 2D screen-space covariance matrix
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	// if (idx == 0){
	// 	printf("After COV 2D\n");
	// }


	// // Invert covariance (EWA algorithm)
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f){
		return;
	}

	if (idx == 0){
		printf("After Det\n");
	}


	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// printf("Cmpute EXTENT\n");
	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;

	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0){
		return;
	}

	// // if (idx == 0) { // Only thread with rank 0 prints
    // //     printf("Thread idx!!!!!!!!!!!!!!!!!!!!!!!\n");
    // // }



	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	// printf("ENTERING COMPUTER COLOR\n");
	if (idx == 0){
		printf("BEFORE COMPUTE COLOR\n");
	}

	// FINE UP TO HERE


	if (colors_precomp == nullptr)
	{	
		if (idx > 1990){
			printf("BADDDDD idx: %d\n", idx);
		}
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}



	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ range)
	// const uint32_t* __restrict__ point_list,
	// int W, int H,
	// const float2* __restrict__ points_xy_image,
	// const float* __restrict__ depths,
	// const float* __restrict__ features,
	// const float4* __restrict__ conic_opacity,
	// float* __restrict__ final_T,
	// uint32_t* __restrict__ n_contrib,
	// const float* __restrict__ bg_color,
	// float* __restrict__ out_color,
	// float* __restrict__ out_depth)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
		return;
        // printf("INSIDE RENDER KERNEL ++++++++++++++++++++++++++++++++++\n");
    }
	// // Identify current tile and associated min/max pixel range.
	// auto block = cg::this_thread_block();
	// uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	// uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	// uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	// uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	// uint32_t pix_id = W * pix.y + pix.x;
	// float2 pixf = { (float)pix.x, (float)pix.y };

	// // Check if this thread is associated with a valid pixel or outside.
	// bool inside = pix.x < W&& pix.y < H;
	// // Done threads can help with fetching, but don't rasterize
	// bool done = !inside;

	// // Load start/end range of IDs to process in bit sorted list.
	// uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	// const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	// int toDo = range.y - range.x;

	// // Allocate storage for batches of collectively fetched data.
	// __shared__ int collected_id[BLOCK_SIZE];
	// __shared__ float2 collected_xy[BLOCK_SIZE];
	// __shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	// __shared__ float collected_depth[BLOCK_SIZE]; // NEW 

	// // Initialize helper variables
	// float T = 1.0f;
	// float depth = 0.f; // NEW
	// uint32_t contributor = 0;
	// uint32_t last_contributor = 0;
	// float C[CHANNELS] = { 0 };

	// // Iterate over batches until all done or range is complete
	// for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	// {
	// 	// End if entire block votes that it is done rasterizing
	// 	int num_done = __syncthreads_count(done);
	// 	if (num_done == BLOCK_SIZE)
	// 		break;

	// 	// Collectively fetch per-Gaussian data from global to shared
	// 	int progress = i * BLOCK_SIZE + block.thread_rank();
	// 	if (range.x + progress < range.y)
	// 	{
	// 		int coll_id = point_list[range.x + progress];
	// 		collected_id[block.thread_rank()] = coll_id;
	// 		collected_xy[block.thread_rank()] = points_xy_image[coll_id];
	// 		collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
	// 		collected_depth[block.thread_rank()] = depths[coll_id]; // NEW
	// 	}
	// 	block.sync();

	// 	// Iterate over current batch
	// 	for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
	// 	{
	// 		// Keep track of current position in range
	// 		contributor++;

	// 		// Resample using conic matrix (cf. "Surface 
	// 		// Splatting" by Zwicker et al., 2001)
	// 		float2 xy = collected_xy[j];
	// 		float2 d = { xy.x - pixf.x, xy.y - pixf.y };
	// 		float4 con_o = collected_conic_opacity[j];
	// 		float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
	// 		if (power > 0.0f)
	// 			continue;

	// 		// Eq. (2) from 3D Gaussian splatting paper.
	// 		// Obtain alpha by multiplying with Gaussian opacity
	// 		// and its exponential falloff from mean.
	// 		// Avoid numerical instabilities (see paper appendix). 
	// 		float alpha = min(0.99f, con_o.w * exp(power));
	// 		if (alpha < 1.0f / 255.0f)
	// 			continue;
	// 		float test_T = T * (1 - alpha);
	// 		if (test_T < 0.0001f)
	// 		{
	// 			done = true;
	// 			continue;
	// 		}

	// 		// Eq. (3) from 3D Gaussian splatting paper.
	// 		for (int ch = 0; ch < CHANNELS; ch++)
	// 			C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

	// 		depth += collected_depth[j] * alpha * T; // NEW
	// 		T = test_T;

	// 		// Keep track of last range entry to update this
	// 		// pixel.
	// 		last_contributor = contributor;
	// 	}
	// }

	// // All threads that treat valid pixel write out their final
	// // rendering data to the frame and auxiliary buffers.
	// // printf("PRINTINGGGGGGG--------------------------");
	// if (inside)
	// {	
	// 	// printf("INSIDE--------------------------\n");
	// 	if (isnan(T) || isinf(T))
    //     printf("NaN/Inf detected in T: pix_id=%d, T=%f\n", pix_id, T);

	// 	if (isnan(depth) || isinf(depth))
	// 		printf("NaN/Inf detected in depth: pix_id=%d, depth=%f\n", pix_id, depth);

	// 	for (int ch = 0; ch < CHANNELS; ch++) {
	// 		if (isnan(C[ch]) || isinf(C[ch]))
	// 			printf("NaN/Inf detected in C[%d]: pix_id=%d, C=%f\n", ch, pix_id, C[ch]);
	// 	}
	// 	final_T[pix_id] = T;
	// 	n_contrib[pix_id] = last_contributor;
	// 	out_depth[pix_id] = depth; // NEW
	// 	for (int ch = 0; ch < CHANNELS; ch++){
	// 		out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
	// 		if (isnan(out_color[ch * H * W + pix_id]) || isinf(out_color[ch * H * W + pix_id]))
	// 			printf("NaN/Inf detected in out_color[%d]: pix_id=%d, out_color=%f\n", ch, pix_id, out_color[ch * H * W + pix_id]);
	// 	}	
	// }
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* depths,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	float* out_depth)
{
	printf("NORMAL RENDER +_+_+_++_+_+_+_+_++\n");

	// Allocate and copy to device memory

	// Check if host variables are NULL or have unexpected values before copying

	if (!ranges) { printf("ERROR: ranges is NULL\n"); }
	else { printf("Host ranges[0]: (%u, %u)\n", ranges[0].x, ranges[0].y); }

	if (!point_list) { printf("ERROR: point_list is NULL\n"); }
	else { printf("Host point_list[0]: %u\n", point_list[0]); }

	if (!means2D) { printf("ERROR: means2D is NULL\n"); }
	else { printf("Host means2D[0]: (%f, %f)\n", means2D[0].x, means2D[0].y); }

	if (!depths) { printf("ERROR: depths is NULL\n"); }
	else { printf("Host depths[0]: %f\n", depths[0]); }

	if (!colors) { printf("ERROR: colors is NULL\n"); }
	else { printf("Host colors[0]: %f\n", colors[0]); }

	if (!conic_opacity) { printf("ERROR: conic_opacity is NULL\n"); }
	else { printf("Host conic_opacity[0]: (%f, %f, %f, %f)\n", 
				conic_opacity[0].x, conic_opacity[0].y, conic_opacity[0].z, conic_opacity[0].w); }

	if (!final_T) { printf("ERROR: final_T is NULL\n"); }
	else { printf("Host final_T[0]: %f\n", final_T[0]); }

	if (!n_contrib) { printf("ERROR: n_contrib is NULL\n"); }
	else { printf("Host n_contrib[0]: %u\n", n_contrib[0]); }

	if (!bg_color) { printf("ERROR: bg_color is NULL\n"); }
	else { printf("Host bg_color: (%f, %f, %f)\n", bg_color[0], bg_color[1], bg_color[2]);} 

	if (!out_color) { printf("ERROR: out_color is NULL\n"); }
	else { printf("Host out_color[0]: %u\n", out_color[0]); }

	if (!out_depth) { printf("ERROR: out_depth is NULL\n"); }
	else { printf("Host out_depth[0]: %u\n", out_depth[0]); }




	// Ranges (Same size as `ranges` in host memory)
	uint2* d_ranges;
	size_t ranges_size = sizeof(uint2) * (sizeof(ranges) / sizeof(ranges[0]));
	cudaMalloc((void**)&d_ranges, ranges_size);
	cudaMemcpy(d_ranges, ranges, ranges_size, cudaMemcpyHostToDevice);

	// Point List (Same size as `point_list`)
	uint32_t* d_point_list;
	size_t point_list_size = sizeof(uint32_t) * (sizeof(point_list) / sizeof(point_list[0]));
	cudaMalloc((void**)&d_point_list, point_list_size);
	cudaMemcpy(d_point_list, point_list, point_list_size, cudaMemcpyHostToDevice);

	// Means2D (Same size as `means2D`)
	float2* d_means2D;
	size_t means2D_size = sizeof(float2) * (sizeof(means2D) / sizeof(means2D[0]));
	cudaMalloc((void**)&d_means2D, means2D_size);
	cudaMemcpy(d_means2D, means2D, means2D_size, cudaMemcpyHostToDevice);

	// Depths (Same size as `depths`)
	float* d_depths;
	size_t depths_size = sizeof(float) * (sizeof(depths) / sizeof(depths[0]));
	cudaMalloc((void**)&d_depths, depths_size);
	cudaMemcpy(d_depths, depths, depths_size, cudaMemcpyHostToDevice);

	// Colors (Same size as `colors`, scaled by NUM_CHANNELS)
	float* d_colors;
	size_t colors_size = sizeof(float) * NUM_CHANNELS * (sizeof(colors) / sizeof(colors[0]));
	cudaMalloc((void**)&d_colors, colors_size);
	cudaMemcpy(d_colors, colors, colors_size, cudaMemcpyHostToDevice);

	// Conic Opacity (Same size as `conic_opacity`)
	float4* d_conic_opacity;
	size_t conic_opacity_size = sizeof(float4) * (sizeof(conic_opacity) / sizeof(conic_opacity[0]));
	cudaMalloc((void**)&d_conic_opacity, conic_opacity_size);
	cudaMemcpy(d_conic_opacity, conic_opacity, conic_opacity_size, cudaMemcpyHostToDevice);

	// Final_T (Same size as `final_T`)
	float* d_final_T;
	size_t final_T_size = sizeof(float) * (sizeof(final_T) / sizeof(final_T[0]));
	cudaMalloc((void**)&d_final_T, final_T_size);
	cudaMemcpy(d_final_T, final_T, final_T_size, cudaMemcpyHostToDevice);

	// Contribution Counter (Same size as `n_contrib`)
	uint32_t* d_n_contrib;
	size_t n_contrib_size = sizeof(uint32_t) * (sizeof(n_contrib) / sizeof(n_contrib[0]));
	cudaMalloc((void**)&d_n_contrib, n_contrib_size);
	cudaMemcpy(d_n_contrib, n_contrib, n_contrib_size, cudaMemcpyHostToDevice);

	// Background Color (Always 3 floats for RGB)
	float* d_bg_color;
	cudaMalloc((void**)&d_bg_color, 3 * sizeof(float));
	cudaMemcpy(d_bg_color, bg_color, 3 * sizeof(float), cudaMemcpyHostToDevice);

	// Output Color Buffer (Same size as `out_color`, scaled by NUM_CHANNELS)
	float* d_out_color;
	size_t out_color_size = sizeof(float) * NUM_CHANNELS * (sizeof(out_color) / sizeof(out_color[0]));
	cudaMalloc((void**)&d_out_color, out_color_size);
	cudaMemcpy(d_out_color, out_color, out_color_size, cudaMemcpyHostToDevice);

	// Output Depth Buffer (Same size as `out_depth`)
	float* d_out_depth;
	size_t out_depth_size = sizeof(float) * (sizeof(out_depth) / sizeof(out_depth[0]));
	cudaMalloc((void**)&d_out_depth, out_depth_size);
	cudaMemcpy(d_out_depth, out_depth, out_depth_size, cudaMemcpyHostToDevice);


	// Launch CUDA Kernel
	renderCUDA<NUM_CHANNELS><<<grid, block>>>(
		d_ranges
		// d_point_list,
		// W, H,
		// d_means2D,
		// d_depths,
		// d_colors,
		// d_conic_opacity,
		// d_final_T,
		// d_n_contrib,
		// d_bg_color,
		// d_out_color,
		// d_out_depth
	);


	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA Kernel launch failed: %s\n", cudaGetErrorString(err));
	}
	cudaDeviceSynchronize();  // Ensure the kernel completes
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA Error after synchronize: %s\n", cudaGetErrorString(err));
	} else {
		printf("CUDA Kernel launched and synchronized successfully.\n");
	}
}

void FORWARD::preprocess(int P, int D, int M,
    const float* means3D,
    const glm::vec3* scales,
    const float scale_modifier,
    const glm::vec4* rotations,
    const float* opacities,
    const float* shs,
    bool* clamped,
    const float* cov3D_precomp,
    const float* colors_precomp,
    const float* viewmatrix,
    const float* projmatrix,
    const glm::vec3* cam_pos,
    const int W, int H,
    const float focal_x, float focal_y,
    const float tan_fovx, float tan_fovy,
    int* radii,
    float2* means2D,
    float* depths,
    float* cov3Ds,
    float* rgb,
    float4* conic_opacity,
    const dim3 grid,
    uint32_t* tiles_touched,
    bool prefiltered)
{
    printf("ENTERING PREPROCESS\n");

    // Ensure P is valid
    if (P <= 0) {
        printf("ERROR: P is non-positive (%d), skipping kernel launch.\n", P);
        return;
    }

    // Check for CUDA errors before allocation
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error before preprocess cuda: %s\n", cudaGetErrorString(err));
    }

    float* temp_colors_precomp = nullptr;
    float* temp_cov3D_precomp = nullptr;

    // Allocate memory if needed
    if (colors_precomp == nullptr) {
        printf("Allocating colors_precomp.\n");
        err = cudaMalloc((void**)&temp_colors_precomp, P * 3 * sizeof(float));
        if (err != cudaSuccess) {
            printf("cudaMalloc failed for colors_precomp: %s\n", cudaGetErrorString(err));
        }
        cudaMemset(temp_colors_precomp, 0, P * 3 * sizeof(float));
    } else {
        temp_colors_precomp = (float*)colors_precomp;
    }

    if (cov3D_precomp == nullptr) {
        printf("Allocating cov3D_precomp.\n");
        err = cudaMalloc((void**)&temp_cov3D_precomp, P * 6 * sizeof(float));
        if (err != cudaSuccess) {
            printf("cudaMalloc failed for cov3D_precomp: %s\n", cudaGetErrorString(err));
        }
        cudaMemset(temp_cov3D_precomp, 0, P * 6 * sizeof(float));
    } else {
        temp_cov3D_precomp = (float*)cov3D_precomp;
    }

    // Validate kernel launch parameters
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int blocks = (P + 255) / 256;
    int threads = 256;

    printf("Launching kernel with %d blocks and %d threads per block.\n", blocks, threads);
    if (blocks > prop.maxGridSize[0] || threads > prop.maxThreadsPerBlock) {
        printf("ERROR: Exceeding kernel launch limits! Grid: %d, Threads: %d\n", 
                blocks, threads);
        return;
    }

	printf("viewmatrix (host): %p\n", viewmatrix);
	if (!viewmatrix) {
		printf("ERROR: viewmatrix is NULL before kernel launch!\n");
		return;
	}

	printf("Host: grid = (%d, %d, %d)\n", grid.x, grid.y, grid.z);

	for (int i = 0; i < min(P, 10); i++) {
		printf("Before Kernel, tiles_touched[%d] = %u\n", i, tiles_touched[i]);
	}

	// ViewMatrix
	float* d_viewmatrix;
	cudaMalloc((void**)&d_viewmatrix, 16 * sizeof(float));
	cudaMemcpy(d_viewmatrix, viewmatrix, 16 * sizeof(float), cudaMemcpyHostToDevice);

	// ProjMatrix
	float* d_projmatrix;
	cudaMalloc((void**)&d_projmatrix, 16 * sizeof(float));
	cudaMemcpy(d_projmatrix, projmatrix, 16 * sizeof(float), cudaMemcpyHostToDevice);

	// Radii
	int* d_radii;
	cudaMalloc((void**)&d_radii, P * sizeof(int));
	cudaMemcpy(d_radii, radii, P * sizeof(int), cudaMemcpyHostToDevice);

	// Depths
	float* d_depths;
	cudaMalloc((void**)&d_depths, P * sizeof(float));
	cudaMemcpy(d_depths, depths, P * sizeof(float), cudaMemcpyHostToDevice);

	// Tiles Touched
	uint32_t* d_tiles_touched;
    cudaMalloc((void**)&d_tiles_touched, P * sizeof(uint32_t));
    cudaMemcpy(d_tiles_touched, tiles_touched, P * sizeof(uint32_t), cudaMemcpyHostToDevice);

	// Conic Opacity
	float4* d_conic_opacity;
    cudaMalloc((void**)&d_conic_opacity, P * sizeof(float4));
	cudaMemcpy(d_conic_opacity, conic_opacity, P * sizeof(float4), cudaMemcpyHostToDevice);

	// Means2D
	float2* d_means2D;
    cudaMalloc((void**)&d_means2D, P * sizeof(float2));
	cudaMemcpy(d_means2D, means2D, P * sizeof(float2), cudaMemcpyHostToDevice);


    preprocessCUDA<NUM_CHANNELS><<<blocks, threads>>>(
        P, D, M,
        means3D,
        scales,
        scale_modifier,
        rotations,
        opacities,
        shs,
        clamped,
        temp_cov3D_precomp,
        temp_colors_precomp,
        d_viewmatrix, 
        d_projmatrix,
        cam_pos,
        W, H,
        tan_fovx, tan_fovy,
        focal_x, focal_y,
        d_radii,
        d_means2D,
        d_depths,
        cov3Ds,
        rgb,
        d_conic_opacity,
        grid,
        d_tiles_touched,
        prefiltered
    );

    // Synchronize and check for errors
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error after preprocessCUDA kernel execution: %s\n", cudaGetErrorString(err));
    }


	err = cudaMemcpy(radii, d_radii, P * sizeof(int), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		printf("🚨 cudaMemcpy radii failed: %s\n", cudaGetErrorString(err));
	}
	
	err = cudaMemcpy(depths, d_depths, P * sizeof(float), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		printf("🚨 cudaMemcpy depths failed: %s\n", cudaGetErrorString(err));
	}
	
	err = cudaMemcpy(tiles_touched, d_tiles_touched, P * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		printf("🚨 cudaMemcpy tiles_touched failed: %s\n", cudaGetErrorString(err));
	}

	for (int i = 0; i < min(P, 10); i++) {
		printf("After Kernel, tiles_touched[%d] = %u\n", i, tiles_touched[i]);
	}

	
	err = cudaMemcpy(conic_opacity, d_conic_opacity, P * sizeof(float4), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		printf("🚨 cudaMemcpy conic_opacity failed: %s\n", cudaGetErrorString(err));
	}
	
	err = cudaMemcpy(means2D, d_means2D, P * sizeof(float2), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		printf("🚨 cudaMemcpy means2D failed: %s\n", cudaGetErrorString(err));
	}

	
	// Free GPU memory
	cudaFree(d_viewmatrix);
	cudaFree(d_projmatrix);
	cudaFree(d_radii);
	cudaFree(d_depths);
	cudaFree(d_tiles_touched);
	cudaFree(d_conic_opacity);
	cudaFree(d_means2D);

}
