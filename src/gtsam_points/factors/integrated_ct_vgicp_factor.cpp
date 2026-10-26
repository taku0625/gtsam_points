// SPDX-License-Identifier: MIT
#include <gtsam_points/factors/integrated_ct_vgicp_factor.hpp>

#include <limits>
#include <gtsam_points/factors/impl/integrated_ct_icp_factor_impl.hpp>
#include <gtsam_points/factors/impl/integrated_ct_gicp_factor_impl.hpp>

namespace gtsam_points {
namespace {

// Return the map's encoded point index so that CT-GICP reads the voxel mean
// and covariance through the existing GaussianVoxelMapCPU frame traits.
class VoxelCorrespondenceSearch : public NearestNeighborSearch {
public:
  explicit VoxelCorrespondenceSearch(const GaussianVoxelMapCPU::ConstPtr& target) : target(target) {}

  size_t knn_search(const double* pt, size_t k, size_t* indices, double* sq_dists, double max_sq_dist) const override {
    if (k == 0) {
      return 0;
    }

    const Eigen::Vector4d query(pt[0], pt[1], pt[2], 1.0);
    const int voxel_id = target->lookup_voxel_index(target->voxel_coord(query));
    if (voxel_id < 0) {
      return 0;
    }

    const double sq_dist = (query - target->lookup_voxel(voxel_id).mean).squaredNorm();
    if (sq_dist > max_sq_dist) {
      return 0;
    }

    indices[0] = target->calc_index(voxel_id, 0);
    sq_dists[0] = sq_dist;
    return 1;
  }

private:
  GaussianVoxelMapCPU::ConstPtr target;
};

}  // namespace

IntegratedCT_VGICPFactor::IntegratedCT_VGICPFactor(
  gtsam::Key source_t0_key,
  gtsam::Key source_t1_key,
  const std::shared_ptr<const GaussianVoxelMapCPU>& target,
  const PointCloud::ConstPtr& source)
: IntegratedCT_GICPFactor_<GaussianVoxelMapCPU, PointCloud>(
    source_t0_key, source_t1_key, target, source, std::make_shared<VoxelCorrespondenceSearch>(target)) {
  // DIRECT1 is bounded by voxel membership, not nearest-neighbor distance.
  set_max_correspondence_distance(std::numeric_limits<double>::infinity());
}

IntegratedCT_VGICPFactor::~IntegratedCT_VGICPFactor() {}

double IntegratedCT_VGICPFactor::error(const gtsam::Values& values) const {
  update_poses(values);
  if (correspondences.size() != source->size()) {
    update_correspondences();
  }

  // GTSAM's nonlinear error is 1/2 r^T W r. Keep that convention independent
  // of the CT-GICP implementation's error scaling in different base versions.
  double sum_errors = 0.0;
#pragma omp parallel for reduction(+ : sum_errors) schedule(guided, 8) num_threads(num_threads)
  for (int i = 0; i < source->size(); i++) {
    const long index = correspondences[i];
    if (index < 0) {
      continue;
    }
    const Eigen::Vector4d residual = source_poses[time_indices[i]].matrix() * source->points[i] - target->point(index);
    sum_errors += 0.5 * residual.dot(mahalanobis[i] * residual);
  }
  return sum_errors;
}

gtsam::GaussianFactor::shared_ptr IntegratedCT_VGICPFactor::linearize(const gtsam::Values& values) const {
  // Reuse CT-GICP's endpoint Jacobians and normal equations. A HessianFactor
  // also applies 1/2 to its constant term, so store r^T W r there.
  const auto info = IntegratedCT_GICPFactor_<GaussianVoxelMapCPU, PointCloud>::linearize(values)->augmentedInformation();
  return gtsam::GaussianFactor::shared_ptr(new gtsam::HessianFactor(
    keys_[0], keys_[1],
    info.block<6, 6>(0, 0), info.block<6, 6>(0, 6), info.block<6, 1>(0, 12),
    info.block<6, 6>(6, 6), info.block<6, 1>(6, 12), 2.0 * error(values)));
}

// Export the nonvirtual CT utilities (including deskewing) for this target type.
template class IntegratedCT_ICPFactor_<GaussianVoxelMapCPU, PointCloud>;
template class IntegratedCT_GICPFactor_<GaussianVoxelMapCPU, PointCloud>;

}  // namespace gtsam_points
