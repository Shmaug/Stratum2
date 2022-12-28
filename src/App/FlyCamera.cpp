#include "FlyCamera.hpp"

#include "Scene.hpp"
#include "Inspector.hpp"

#include <Core/Profiler.hpp>
#include <Core/Swapchain.hpp>

#include <imgui/imgui.h>

namespace stm2 {

FlyCamera::FlyCamera(Node& node) : mNode(node) {
	if (shared_ptr<Inspector> inspector = mNode.root()->findDescendant<Inspector>())
		inspector->setInspectCallback<FlyCamera>();
}

void FlyCamera::drawGui() {
	ImGui::DragFloat("Move Speed", &mMoveSpeed, 0.1f, 0);
	if (mMoveSpeed < 0) mMoveSpeed = 0;
	ImGui::DragFloat("Rotate Speed", &mRotateSpeed, 0.001f, 0);
	if (mRotateSpeed < 0) mRotateSpeed = 0;

	if (ImGui::DragFloat2("Rotation", mRotation.data(), 0.01f, 0))
		mRotation.x() = clamp(mRotation.x(), -((float)M_PI) / 2, ((float)M_PI) / 2);
}

void FlyCamera::update(const float deltaTime) {
	ProfilerScope ps("FlyCamera::update");

	const float fwd = mNode.getComponent<Camera>()->mProjection.mNearPlane > 0 ? 1 : -1;

	TransformData& transform = *mNode.findAncestor<TransformData>();

	const ImGuiIO& io = ImGui::GetIO();

	if (!io.WantCaptureMouse) {
		if (ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
			if (io.MouseWheel != 0)
				mMoveSpeed *= (1 + io.MouseWheel / 8);

			mRotation[1] += io.MouseDelta.x * fwd * mRotateSpeed;
			mRotation[0] = clamp(mRotation[0] + io.MouseDelta.y * mRotateSpeed, -((float)M_PI) / 2, ((float)M_PI) / 2);

			const quatf r = qmul(
				quatf::angleAxis(mRotation.y(), float3(0, 1, 0)),
				quatf::angleAxis(mRotation.x(), float3(fwd, 0, 0)) );
			transform.m.block<3, 3>(0, 0) = Eigen::Quaternionf(r.w, r.xyz[0], r.xyz[1], r.xyz[2]).matrix();
		}
	}
	if (!io.WantCaptureKeyboard) {
		float3 mv = float3(0, 0, 0);
		if (ImGui::IsKeyDown(ImGuiKey_D))     mv += float3(1, 0, 0);
		if (ImGui::IsKeyDown(ImGuiKey_A))     mv += float3(-1, 0, 0);
		if (ImGui::IsKeyDown(ImGuiKey_W))     mv += float3(0, 0, fwd);
		if (ImGui::IsKeyDown(ImGuiKey_S))     mv += float3(0, 0, -fwd);
		if (ImGui::IsKeyDown(ImGuiKey_Space)) mv += float3(0, 1, 0);
		if (ImGui::IsKeyDown(ImGuiKey_C))     mv += float3(0, -1, 0);
		if (!mv.isZero())
			transform = tmul(transform, TransformData(mv * mMoveSpeed * deltaTime, quatf::identity(), float3::Ones()));
	}
}

}
