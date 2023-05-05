#include "Inspector.hpp"
#include "Scene.hpp"
#include "Gui.hpp"
#include <Core/Instance.hpp>
#include <Core/Swapchain.hpp>
#include <Core/Window.hpp>
#include <Core/Profiler.hpp>

#include <imgui/imgui.h>
#include <ImGuizmo.h>

namespace stm2 {

// dont allow ceration/deletion of core components
static unordered_set<type_index> gProtectedComponents = {
	typeid(Window),
	typeid(Instance),
	typeid(Device),
	typeid(Swapchain),
	typeid(Scene),
	typeid(Gui),
	typeid(Inspector)
};

Inspector::Inspector(Node& node) : mNode(node) {
	setInspectCallback<Device>();
	setInspectCallback<Instance>();
	setInspectCallback<Swapchain>();
	setInspectCallback<Window>();
}

// scene graph node inspector
bool Inspector::drawNodeGui(Node& n) {
	ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnDoubleClick | ImGuiTreeNodeFlags_OpenOnArrow;
	auto sel = selected();
	if (&n == sel.get()) flags |= ImGuiTreeNodeFlags_Selected;
	if (n.children().empty()) flags |= ImGuiTreeNodeFlags_Leaf;

	// open nodes above selected node
	if (sel && n.isDescendant(*sel))
		ImGui::SetNextItemOpen(true, ImGuiCond_Once);


	// tree menu item
	ImGui::PushID(&n);
	const bool open = ImGui::TreeNodeEx(n.name().c_str(), flags);
	ImGui::PopID();

	if (ImGui::BeginDragDropSource()) {
		const Node* payload = &n;
		ImGui::SetDragDropPayload("SceneNode", &payload, sizeof(Node*));
		ImGui::Text("%s", n.name().c_str());
		ImGui::EndDragDropSource();
	}
	if (ImGui::BeginDragDropTarget()) {
		if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("SceneNode")) {
			Node* nodeptr = *(Node**)payload->Data;
			if (nodeptr) {
				n.addChild(nodeptr->getPtr());
				if (auto s = n.findAncestor<Scene>())
					s->markDirty();
			}
		}
		if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("SceneComponent")) {
			const auto&[nodeptr, type] = *(pair<Node*, type_index>*)payload->Data;
			if (nodeptr && !n.hasComponent(type)) {
				shared_ptr<void> component = nodeptr->getComponent(type);
				nodeptr->removeComponent(type);
				n.addComponent(type, component);
				if (auto s = n.findAncestor<Scene>())
					s->markDirty();
			}
		}
		ImGui::EndDragDropTarget();
	}

	bool erase = false;

	// context menu
	if (ImGui::BeginPopupContextItem()) {
		if (ImGui::Selectable("Add component", false, ImGuiSelectableFlags_DontClosePopups)) {
			ImGui::OpenPopup("Add component");
		}
		if (ImGui::Selectable("Add child", false, ImGuiSelectableFlags_DontClosePopups)) {
			ImGui::OpenPopup("Add node");
		}
		bool canDelete = true;
		for (type_index c : n.components()) {
			if (gProtectedComponents.contains(c)) {
				canDelete = false;
				break;
			}
		}
		if (ImGui::Selectable("Delete", false, canDelete ? 0 : ImGuiSelectableFlags_Disabled)) {
			erase = true;
		}

		// add component dialog
		if (ImGui::BeginPopup("Add component")) {
			for (auto[type, ctor] : mInspectorConstructFns) {
				if (!gProtectedComponents.contains(type) && ImGui::Selectable(type.name(), n.hasComponent(type) ? ImGuiSelectableFlags_Disabled : 0)) {
					n.addComponent(type, ctor(n));
					ImGui::CloseCurrentPopup();
				}
			}
			ImGui::EndPopup();
		}

		// add child dialog
		if (ImGui::BeginPopup("Add node")) {
			mInputChildName.resize(64);
			ImGui::InputText("Child name", mInputChildName.data(), mInputChildName.size());
			if (ImGui::Button("Done")) {
				n.addChild(mInputChildName);
				ImGui::CloseCurrentPopup();
			}
			ImGui::EndPopup();
		}

		ImGui::EndPopup();
	}

	if (open) {
		// select node if treenode is clicked
		if (ImGui::IsItemClicked())
			select(n.getPtr());

		unordered_set<Node*> toErase;

		// draw children
		for (const shared_ptr<Node>& c : n.children())
			if (drawNodeGui(*c))
				toErase.emplace(c.get());

		if (auto s = n.findAncestor<Scene>(); s && !toErase.empty())
			s->markDirty();

		for (Node* c : toErase) {
			c->removeParent();
			if (selected().get() == c)
				select(nullptr);
		}

		ImGui::TreePop();
	}
	return erase;
}

void Inspector::draw() {
	ProfilerScope ps("Inspector::draw");

	if (ImGui::Begin("Node Graph")) {
		const float s = ImGui::GetStyle().IndentSpacing;
		ImGui::GetStyle().IndentSpacing = s/2;
		drawNodeGui(*mNode.root());
		ImGui::GetStyle().IndentSpacing = s;
	}
	ImGui::End();

	if (ImGui::Begin("Inspector")) {
		if (auto sel = selected()) {
			ImGui::PushID(sel.get());

			// inspect selected node

			ImGui::PushFont(Gui::gHeaderFont);
			ImGui::Text("%s", sel->name().c_str());
			ImGui::PopFont();

			// components

			type_index to_erase = typeid(nullptr_t);
			for (type_index type : sel->components()) {
				ImGui::PushID(type.hash_code());

				ImGui::PushFont(Gui::gHeaderFont);
				ImGui::SetWindowFontScale(0.85f);
				const bool open = ImGui::CollapsingHeader(type.name(), ImGuiTreeNodeFlags_DefaultOpen);
				ImGui::SetWindowFontScale(1.0f);
				ImGui::PopFont();

				if (!gProtectedComponents.contains(type) && ImGui::BeginDragDropSource()) {
					const pair<Node*, type_index> payload { sel.get(), type };
					ImGui::SetDragDropPayload("SceneComponent", &payload, sizeof(payload));
					ImGui::Text("%s", type.name());
					ImGui::EndDragDropSource();
				}

				// context menu
				if (ImGui::BeginPopupContextItem()) {
					if (ImGui::Selectable("Delete", false, gProtectedComponents.contains(type) ? ImGuiSelectableFlags_Disabled : 0)) {
						to_erase = type;
					}
					ImGui::EndPopup();
				}

				if (open) {
					if (auto it = mInspectorGuiFns.find(type); it != mInspectorGuiFns.end()) {
						// draw gui for component
						ImGui::Indent();
						it->second(*sel);
						ImGui::Unindent();
					}
				}

				ImGui::PopID();
			}
			if (to_erase != typeid(nullptr_t)) {
				sel->removeComponent(to_erase);
				if (auto scene = sel->findAncestor<Scene>())
					scene->markDirty();
			}

			ImGui::PopID();
		} else
			ImGui::Text("%s", "Select a node to inspect");
	}
	ImGui::End();

	if (!mPinned.empty()) {
		if (ImGui::Begin("Pinned")) {
			for (auto it = mPinned.begin(); it != mPinned.end();) {
				const auto&[label, weakptr, drawFn] = it->second;
				if (const auto node = weakptr.lock()) {
					ImGui::PushID(node.get());
					if (ImGui::Button("x")) {
						it = mPinned.erase(it);
						ImGui::PopID();
						continue;
					}
					ImGui::SameLine();
					if (ImGui::CollapsingHeader(label.c_str())) {
						if (drawFn.index() == 0)
							get<0>(drawFn)();
						else
							get<1>(drawFn)(*node);
					}
					ImGui::PopID();
				}
				it++;
			}
		}
		ImGui::End();
	}
}

}