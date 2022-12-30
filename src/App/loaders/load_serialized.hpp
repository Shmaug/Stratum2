#pragma once

#include <App/Scene.hpp>

namespace stm2 {

Mesh loadSerialized(CommandBuffer& commandBuffer, const filesystem::path& filename, int shape_index);

}