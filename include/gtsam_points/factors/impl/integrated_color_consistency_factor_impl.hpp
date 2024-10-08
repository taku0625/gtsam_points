// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_points/factors/integrated_color_consistency_factor.hpp>

#include <vector>
#include <gtsam_points/ann/nearest_neighbor_search.hpp>

namespace gtsam_points {

template <typename TargetFrame, typename SourceFrame, typename IntensityGradients>
IntegratedColorConsistencyFactor_<TargetFrame, SourceFrame, IntensityGradients>::IntegratedColorConsistencyFactor_(
  gtsam::Key target_key,
  gtsam::Key source_key,
  const std::shared_ptr<const TargetFrame>& target,
  const std::shared_ptr<const SourceFrame>& source,
  const std::shared_ptr<const NearestNeighborSearch>& target_tree,
  const std::shared_ptr<const IntensityGradients>& target_gradients)
: IntegratedMatchingCostFactor(target_key, source_key),
  num_threads(1),
  max_correspondence_distance_sq(1.0),
  photometric_term_weight(1.0),
  correspondence_update_tolerance_rot(0.0),
  correspondence_update_tolerance_trans(0.0),
  target(target),
  source(source),
  target_tree(target_tree),
  target_gradients(target_gradients) {
  //
  if (!frame::has_points(*target) || !frame::has_normals(*target) || !frame::has_intensities(*target)) {
    std::cerr << "error: target frame doesn't have required attributes for colored_gicp" << std::endl;
    abort();
  }

  if (!frame::has_points(*source) || !frame::has_intensities(*source)) {
    std::cerr << "error: source frame doesn't have required attributes for colored_gicp" << std::endl;
    abort();
  }
}

template <typename TargetFrame, typename SourceFrame, typename IntensityGradients>
IntegratedColorConsistencyFactor_<TargetFrame, SourceFrame, IntensityGradients>::IntegratedColorConsistencyFactor_(
  const gtsam::Pose3& fixed_target_pose,
  gtsam::Key source_key,
  const std::shared_ptr<const TargetFrame>& target,
  const std::shared_ptr<const SourceFrame>& source,
  const std::shared_ptr<const NearestNeighborSearch>& target_tree,
  const std::shared_ptr<const IntensityGradients>& target_gradients)
: IntegratedMatchingCostFactor(fixed_target_pose, source_key),
  num_threads(1),
  max_correspondence_distance_sq(1.0),
  photometric_term_weight(1.0),
  correspondence_update_tolerance_rot(0.0),
  correspondence_update_tolerance_trans(0.0),
  target(target),
  source(source),
  target_tree(target_tree),
  target_gradients(target_gradients) {
  //
  if (!frame::has_points(*target) || !frame::has_normals(*target) || !frame::has_intensities(*target)) {
    std::cerr << "error: target frame doesn't have required attributes for colored_gicp" << std::endl;
    abort();
  }

  if (!frame::has_points(*source) || !frame::has_intensities(*source)) {
    std::cerr << "error: source frame doesn't have required attributes for colored_gicp" << std::endl;
    abort();
  }
}

template <typename TargetFrame, typename SourceFrame, typename IntensityGradients>
IntegratedColorConsistencyFactor_<TargetFrame, SourceFrame, IntensityGradients>::~IntegratedColorConsistencyFactor_() {}

template <typename TargetFrame, typename SourceFrame, typename IntensityGradients>
void IntegratedColorConsistencyFactor_<TargetFrame, SourceFrame, IntensityGradients>::update_correspondences(const Eigen::Isometry3d& delta) const {
  bool do_update = true;
  if (correspondences.size() == frame::size(*source) && (correspondence_update_tolerance_trans > 0.0 || correspondence_update_tolerance_rot > 0.0)) {
    Eigen::Isometry3d diff = delta.inverse() * last_correspondence_point;
    double diff_rot = Eigen::AngleAxisd(diff.linear()).angle();
    double diff_trans = diff.translation().norm();
    if (diff_rot < correspondence_update_tolerance_rot && diff_trans < correspondence_update_tolerance_trans) {
      do_update = false;
    }
  }

  correspondences.resize(frame::size(*source));

#pragma omp parallel for num_threads(num_threads) schedule(guided, 8)
  for (int i = 0; i < frame::size(*source); i++) {
    if (do_update) {
      Eigen::Vector4d pt = delta * frame::point(*source, i);
      pt[3] = frame::intensity(*source, i);

      size_t k_index = -1;
      double k_sq_dist = -1;

      size_t num_found = target_tree->knn_search(pt.data(), 1, &k_index, &k_sq_dist, max_correspondence_distance_sq);
      correspondences[i] = (num_found && k_sq_dist < max_correspondence_distance_sq) ? k_index : -1;
    }
  }

  last_correspondence_point = delta;
}

template <typename TargetFrame, typename SourceFrame, typename IntensityGradients>
double IntegratedColorConsistencyFactor_<TargetFrame, SourceFrame, IntensityGradients>::evaluate(
  const Eigen::Isometry3d& delta,
  Eigen::Matrix<double, 6, 6>* H_target,
  Eigen::Matrix<double, 6, 6>* H_source,
  Eigen::Matrix<double, 6, 6>* H_target_source,
  Eigen::Matrix<double, 6, 1>* b_target,
  Eigen::Matrix<double, 6, 1>* b_source) const {
  //
  if (correspondences.size() != frame::size(*source)) {
    update_correspondences(delta);
  }

  double sum_errors_photo = 0.0;

  std::vector<Eigen::Matrix<double, 6, 6>> Hs_target;
  std::vector<Eigen::Matrix<double, 6, 6>> Hs_source;
  std::vector<Eigen::Matrix<double, 6, 6>> Hs_target_source;
  std::vector<Eigen::Matrix<double, 6, 1>> bs_target;
  std::vector<Eigen::Matrix<double, 6, 1>> bs_source;

  if (H_target && H_source && H_target_source && b_target && b_source) {
    Hs_target.resize(num_threads, Eigen::Matrix<double, 6, 6>::Zero());
    Hs_source.resize(num_threads, Eigen::Matrix<double, 6, 6>::Zero());
    Hs_target_source.resize(num_threads, Eigen::Matrix<double, 6, 6>::Zero());
    bs_target.resize(num_threads, Eigen::Matrix<double, 6, 1>::Zero());
    bs_source.resize(num_threads, Eigen::Matrix<double, 6, 1>::Zero());
  }

#pragma omp parallel for num_threads(num_threads) reduction(+ : sum_errors_photo) schedule(guided, 8)
  for (int i = 0; i < frame::size(*source); i++) {
    const long target_index = correspondences[i];
    if (target_index < 0) {
      continue;
    }

    // source atributes
    const auto& mean_A = frame::point(*source, i);
    const double intensity_A = frame::intensity(*source, i);

    // target attributes
    const auto& mean_B = frame::point(*target, target_index);
    const auto& normal_B = frame::normal(*target, target_index);
    const auto& gradient_B = frame::intensity_gradient(*target_gradients, target_index);
    const double intensity_B = frame::intensity(*target, target_index);

    const Eigen::Vector4d transed_A = delta * mean_A;

    // photometric error
    const Eigen::Vector4d projected = transed_A - (transed_A - mean_B).dot(normal_B) * normal_B;
    const Eigen::Vector4d offset = projected - mean_B;
    const double error_photo = intensity_B + gradient_B.dot(offset) - intensity_A;

    sum_errors_photo += 0.5 * error_photo * photometric_term_weight * error_photo;

    if (!H_target) {
      continue;
    }

    int thread_num = 0;
#ifdef _OPENMP
    thread_num = omp_get_thread_num();
#endif

    Eigen::Matrix<double, 4, 6> J_transed_target = Eigen::Matrix<double, 4, 6>::Zero();
    J_transed_target.block<3, 3>(0, 0) = gtsam::SO3::Hat(transed_A.head<3>());
    J_transed_target.block<3, 3>(0, 3) = -Eigen::Matrix3d::Identity();

    Eigen::Matrix<double, 4, 6> J_transed_source = Eigen::Matrix<double, 4, 6>::Zero();
    J_transed_source.block<3, 3>(0, 0) = -delta.linear() * gtsam::SO3::Hat(mean_A.template head<3>());
    J_transed_source.block<3, 3>(0, 3) = delta.linear();

    // photometric error derivatives
    Eigen::Matrix<double, 4, 4> J_projected_transed = Eigen::Matrix4d::Identity() - normal_B * normal_B.transpose();
    J_projected_transed(3, 3) = 0.0;
    const auto& J_offset_transed = J_projected_transed;

    Eigen::Matrix<double, 1, 4> J_ephoto_offset = gradient_B;
    Eigen::Matrix<double, 1, 4> J_ephoto_transed = J_ephoto_offset * J_offset_transed;

    Eigen::Matrix<double, 1, 6> J_ephoto_target = J_ephoto_transed * J_transed_target;
    Eigen::Matrix<double, 1, 6> J_ephoto_source = J_ephoto_transed * J_transed_source;

    Hs_target[thread_num] += J_ephoto_target.transpose() * photometric_term_weight * J_ephoto_target;
    Hs_source[thread_num] += J_ephoto_source.transpose() * photometric_term_weight * J_ephoto_source;
    Hs_target_source[thread_num] += J_ephoto_target.transpose() * photometric_term_weight * J_ephoto_source;
    bs_target[thread_num] += J_ephoto_target.transpose() * photometric_term_weight * error_photo;
    bs_source[thread_num] += J_ephoto_source.transpose() * photometric_term_weight * error_photo;
  }

  if (H_target && H_source && H_target_source && b_target && b_source) {
    H_target->setZero();
    H_source->setZero();
    H_target_source->setZero();
    b_target->setZero();
    b_source->setZero();

    for (int i = 0; i < num_threads; i++) {
      (*H_target) += Hs_target[i];
      (*H_source) += Hs_source[i];
      (*H_target_source) += Hs_target_source[i];
      (*b_target) += bs_target[i];
      (*b_source) += bs_source[i];
    }
  }

  return sum_errors_photo;
}

}  // namespace gtsam_points