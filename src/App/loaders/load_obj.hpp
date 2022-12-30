#pragma once

#include <App/Scene.hpp>

namespace stm2 {

Mesh loadObj(CommandBuffer& commandBuffer, const filesystem::path& filename);

}