#pragma once

#include <Utils/math.hpp>

#include "Node.hpp"

namespace tinyvkpt {

struct FlyCamera {
	Node& mNode;

	float mMoveSpeed = 1;
	float mRotateSpeed = 0.002f;
	float2 mRotation = float2::Zero();

	FlyCamera(Node&);

	void drawGui();
	void update(const float deltaTime);
};

}