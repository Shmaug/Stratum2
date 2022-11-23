#include "Inspector.hpp"
#include "Scene.hpp"
#include <Core/Instance.hpp>
#include <Core/Swapchain.hpp>
#include <Core/Window.hpp>
#include <Core/Profiler.hpp>

#include <imgui/imgui.h>

namespace tinyvkpt {

Inspector::Inspector(Node& node) : mNode(node) {
	setTypeCallback<Device>();
	setTypeCallback<Instance>();
	setTypeCallback<Swapchain>();
	setTypeCallback<Window>();
}

void Inspector::drawNodeGui(Node& n) {
	ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnDoubleClick | ImGuiTreeNodeFlags_OpenOnArrow;
	if (&n == mSelected.get()) flags |= ImGuiTreeNodeFlags_Selected;
	if (n.children().empty()) flags |= ImGuiTreeNodeFlags_Leaf;

	if (mSelected && n.isDescendant(*mSelected))
		ImGui::SetNextItemOpen(true, ImGuiCond_Once);

	ImGui::PushID(&n);
	const bool open = ImGui::TreeNodeEx(n.name().c_str(), flags);
	ImGui::PopID();
	if (open) {
		if (ImGui::IsItemClicked())
			select(n.getPtr());
		for (const shared_ptr<Node>& c : n.children())
			drawNodeGui(*c);
		ImGui::TreePop();
	}
}

void Inspector::draw() {
	ProfilerScope ps("Inspector::update");
	if (ImGui::Begin("Node Graph"))
		drawNodeGui(*mNode.root());
	ImGui::End();

	if (ImGui::Begin("Inspector")) {
		if (mSelected) {
			auto s = mSelected->findAncestor<Scene>();
			if ((s && !mSelected->getComponent<Scene>()) && ImGui::Button("x")) {
				// delete selected node

				s->markDirty();
				if (ImGui::GetIO().KeyAlt) {
					if (const shared_ptr<Node> p = mSelected->parent())
						for (const shared_ptr<Node>& c : mSelected->children())
							p->addChild(c);
				}
				mSelected->removeParent();
				select(nullptr);
			} else {
				// inspect selected node

				ImGui::SameLine();
				ImGui::Text(mSelected->name().c_str());
				ImGui::SetNextItemWidth(40);

				if (!mSelected->getComponent<TransformData>() && ImGui::Button("Add transform"))
					mSelected->makeComponent<TransformData>(float3::Zero(), quatf_identity(), float3::Ones());

				// list components in selected node

				type_index to_erase = typeid(nullptr_t);
				for (type_index type : mSelected->components()) {
					if ((s && !mSelected->getComponent<Scene>())) {
						ImGui::PushID(type.hash_code());
						if (ImGui::Button("x")) to_erase = type;
						ImGui::PopID();
						ImGui::SameLine();
					}
					if (ImGui::CollapsingHeader(type.name())) {
						if (auto it = mInspectorGuiFns.find(type); it != mInspectorGuiFns.end()) {
							// draw gui for component
							ImGui::Indent();
							ImGui::Indent();
							it->second(*mSelected);
							ImGui::Unindent();
							ImGui::Unindent();
						}
					}
				}
				if (to_erase != typeid(nullptr_t)) {
					mSelected->removeComponent(to_erase);
					s->markDirty();
				}
			}
		} else
			ImGui::Text("Select a node to inspect");
	}
	ImGui::End();

	for (const auto[ptr, tpl] : mPinned) {
		const auto&[label, weakptr, drawFn] = tpl;
		if (const auto node = weakptr.lock()) {
			if (ImGui::Begin(label.c_str())) {
				if (drawFn.index() == 0)
					get<0>(drawFn)();
				else
					get<1>(drawFn)(*node);
			}
			ImGui::End();
		}
	}
}

}