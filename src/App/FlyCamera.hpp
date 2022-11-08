#pragma once

#include <Utils/math.hpp>

#include "SceneGraph.hpp"

namespace tinyvkpt {

class Camera;

struct FlyCamera {
	Node& mNode;
	shared_ptr<Camera> mCamera;

	float mMoveSpeed = 1;
	float mRotateSpeed = 0.002f;
	float2 mRotation = float2::Zero();
	bool mMatchWindowRect = true;

	void drawGui();
	void update(const float deltaTime);
};

}