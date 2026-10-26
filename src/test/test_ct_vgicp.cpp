#include <gtest/gtest.h>

#include <gtsam/linear/GaussianFactor.h>
#include <gtsam_points/factors/integrated_ct_vgicp_factor.hpp>
#include <gtsam_points/factors/integrated_vgicp_factor.hpp>
#include <gtsam_points/optimizers/levenberg_marquardt_ext.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>

namespace {

using namespace gtsam_points;

PointCloudCPU::Ptr cloud(const std::vector<Eigen::Vector4d>& points) {
  auto frame = std::make_shared<PointCloudCPU>(points);
  Eigen::Matrix4d cov = Eigen::Matrix4d::Zero();
  cov.diagonal() << 0.03, 0.05, 0.01, 0.0;
  frame->add_covs(std::vector<Eigen::Matrix4d>(points.size(), cov));
  std::vector<double> times(points.size());
  for (size_t i = 0; i < times.size(); i++) {
    times[i] = static_cast<double>(i) / std::max<size_t>(1, times.size() - 1);
  }
  frame->add_times(times);
  return frame;
}

gtsam::Values poses(const gtsam::Pose3& begin = gtsam::Pose3(), const gtsam::Pose3& end = gtsam::Pose3()) {
  gtsam::Values values;
  values.insert(0, begin);
  values.insert(1, end);
  return values;
}

TEST(CTVGICP, MatchesRigidVGICPLinearizationAtEqualEndpointPoses) {
  auto target = std::make_shared<GaussianVoxelMapCPU>(2.0);
  // Two samples per voxel: the mean differs from either original sample.
  target->insert(*cloud({{0.2, 0.3, 0.4, 1}, {0.8, 0.7, 0.6, 1}, {2.2, 0.3, 0.4, 1}, {2.8, 0.7, 0.6, 1}}));
  auto source = cloud({{0.35, 0.4, 0.4, 1}, {2.35, 0.4, 0.4, 1}});
  gtsam::Vector6 tangent;
  tangent << 0.01, -0.02, 0.03, 0.1, -0.05, 0.04;
  const auto pose = gtsam::Pose3::Expmap(tangent);
  auto values = poses(pose, pose);
  IntegratedCT_VGICPFactor ct(0, 1, target, source);
  IntegratedVGICPFactor rigid(gtsam::Pose3(), 0, target, source);
  const auto ct_info = ct.linearize(values)->augmentedInformation();
  const auto rigid_info = rigid.linearize(values)->augmentedInformation();
  const gtsam::Matrix6 combined_H = ct_info.block<6, 6>(0, 0) + ct_info.block<6, 6>(0, 6) +
    ct_info.block<6, 6>(6, 0) + ct_info.block<6, 6>(6, 6);
  const gtsam::Vector6 combined_b = ct_info.block<6, 1>(0, 12) + ct_info.block<6, 1>(6, 12);
  EXPECT_LT((combined_H - rigid_info.block<6, 6>(0, 0)).norm(), 1e-9);
  EXPECT_LT((combined_b - rigid_info.block<6, 1>(0, 6)).norm(), 1e-9);
  EXPECT_GT(ct.error(values), 0.0);
  EXPECT_NEAR(0.5 * ct_info(12, 12), ct.error(values), 1e-9);
}

TEST(CTVGICP, RejectsMissingContainingVoxelEvenWithNearbyPoint) {
  auto target = std::make_shared<GaussianVoxelMapCPU>(1.0);
  target->insert(*cloud({{0.99, 0.5, 0.5, 1}}));
  auto source = cloud({{1.01, 0.5, 0.5, 1}, {-0.01, 0.5, 0.5, 1}});
  IntegratedCT_VGICPFactor factor(0, 1, target, source);
  const auto values = poses();
  EXPECT_DOUBLE_EQ(factor.error(values), 0.0);
  EXPECT_NEAR(factor.linearize(values)->augmentedInformation().norm(), 0.0, 1e-12);
}

TEST(CTVGICP, SameVoxelCorrespondenceHasNoImplicitOneMeterCutoff) {
  auto target = std::make_shared<GaussianVoxelMapCPU>(10.0);
  target->insert(*cloud({{0.1, 0.5, 0.5, 1}}));
  auto source = cloud({{5.1, 0.5, 0.5, 1}});
  IntegratedCT_VGICPFactor factor(0, 1, target, source);
  const auto values = poses();
  EXPECT_GT(factor.error(values), 1.0);
  factor.set_max_correspondence_distance(1.0);
  factor.linearize(values);
  EXPECT_DOUBLE_EQ(factor.error(values), 0.0);
}

TEST(CTVGICP, EndpointGradientMatchesFiniteDifferences) {
  auto target = std::make_shared<GaussianVoxelMapCPU>(2.0);
  target->insert(*cloud({{0.5, 0.5, 0.5, 1}, {2.5, 0.5, 0.5, 1}, {0.5, 2.5, 0.5, 1}, {0.5, 0.5, 2.5, 1}}));
  auto source = cloud({{0.45, 0.55, 0.4, 1}, {2.45, 0.55, 0.4, 1}, {0.45, 2.55, 0.4, 1}, {0.45, 0.55, 2.4, 1}});
  gtsam::Vector6 tangent;
  tangent << 0.01, -0.02, 0.03, 0.05, -0.04, 0.02;
  auto values = poses(gtsam::Pose3::Expmap(tangent), gtsam::Pose3::Expmap(2.0 * tangent));
  IntegratedCT_VGICPFactor factor(0, 1, target, source);
  const auto info = factor.linearize(values)->augmentedInformation();
  const double eps = 1e-6;
  for (int key = 0; key < 2; key++) {
    for (int axis = 0; axis < 6; axis++) {
      gtsam::Vector6 delta = gtsam::Vector6::Zero();
      delta[axis] = eps;
      auto plus = values;
      auto minus = values;
      plus.update(key, values.at<gtsam::Pose3>(key).retract(delta));
      minus.update(key, values.at<gtsam::Pose3>(key).retract(-delta));
      // Keep correspondences and covariance weights fixed, as in CT-GICP LM.
      const double gradient = (factor.error(plus) - factor.error(minus)) / (2.0 * eps);
      EXPECT_NEAR(gradient, -info(key * 6 + axis, 12), 1e-5);
    }
  }
}

TEST(CTVGICP, RecoversContinuousMotionAndDeskews) {
  std::vector<Eigen::Vector4d> target_points;
  for (int x = 0; x < 4; x++) {
    for (int y = 0; y < 4; y++) {
      for (int z = 0; z < 4; z++) {
        target_points.emplace_back(2.0 * x + 0.7, 2.0 * y + 0.7, 2.0 * z + 0.7, 1.0);
      }
    }
  }
  auto target = std::make_shared<GaussianVoxelMapCPU>(2.0);
  target->insert(*cloud(target_points));
  gtsam::Vector6 begin_twist, scan_twist;
  begin_twist << 0.01, -0.01, 0.01, 0.1, -0.1, 0.05;
  scan_twist << 0.01, 0.01, -0.01, 0.1, 0.05, -0.05;
  const auto begin = gtsam::Pose3::Expmap(begin_twist);
  const auto end = begin * gtsam::Pose3::Expmap(scan_twist);
  std::vector<Eigen::Vector4d> source_points;
  for (size_t i = 0; i < target_points.size(); i++) {
    const double t = static_cast<double>(i) / (target_points.size() - 1);
    const auto pose = begin * gtsam::Pose3::Expmap(t * scan_twist);
    source_points.emplace_back(pose.inverse().matrix() * target_points[i]);
  }
  auto source = cloud(source_points);
  IntegratedCT_GICPFactor_<GaussianVoxelMapCPU>::shared_ptr factor(new IntegratedCT_VGICPFactor(0, 1, target, source));
  factor->set_num_threads(2);
  gtsam::NonlinearFactorGraph graph;
  graph.add(factor);
  LevenbergMarquardtExtParams params;
  params.setMaxIterations(30);
  params.setAbsoluteErrorTol(1e-10);
  const auto result = LevenbergMarquardtOptimizerExt(graph, poses(), params).optimize();
  EXPECT_LT(gtsam::Pose3::Logmap(begin.between(result.at<gtsam::Pose3>(0))).norm(), 1e-4);
  EXPECT_LT(gtsam::Pose3::Logmap(end.between(result.at<gtsam::Pose3>(1))).norm(), 1e-4);
  const auto deskewed = factor->deskewed_source_points(result);
  const auto local = factor->deskewed_source_points(result, true);
  for (size_t i = 0; i < deskewed.size(); i++) {
    EXPECT_LT((deskewed[i] - target_points[i]).norm(), 1e-4);
    EXPECT_LT((local[i] - begin.inverse().matrix() * target_points[i]).norm(), 1e-4);
  }
}

}  // namespace
