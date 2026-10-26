// SPDX-License-Identifier: MIT
#pragma once

#include <gtsam/linear/GaussianFactor.h>
#include <gtsam_points/factors/integrated_ct_gicp_factor.hpp>
#include <gtsam_points/types/gaussian_voxelmap_cpu.hpp>

namespace gtsam_points {

/**
 * @brief Continuous-time VGICP against a fixed Gaussian voxel map.
 *
 * Uses the same pose interpolation, Jacobians and deskewing as CT-GICP, but
 * associates each transformed source point with the Gaussian in its containing
 * voxel (DIRECT1), just like IntegratedVGICPFactor. The map must not be modified
 * during optimization. Correspondences are refreshed at each linearization.
 */
class IntegratedCT_VGICPFactor : public IntegratedCT_GICPFactor_<GaussianVoxelMapCPU, PointCloud> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  IntegratedCT_VGICPFactor(
    gtsam::Key source_t0_key,
    gtsam::Key source_t1_key,
    const std::shared_ptr<const GaussianVoxelMapCPU>& target,
    const PointCloud::ConstPtr& source);

  virtual ~IntegratedCT_VGICPFactor() override;

  virtual double error(const gtsam::Values& values) const override;
  virtual gtsam::GaussianFactor::shared_ptr linearize(const gtsam::Values& values) const override;
};

}  // namespace gtsam_points
