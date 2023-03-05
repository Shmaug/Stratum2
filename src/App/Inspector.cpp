#include "Inspector.hpp"
#include "Scene.hpp"
#include <Core/Instance.hpp>
#include <Core/Swapchain.hpp>
#include <Core/Window.hpp>
#include <Core/Profiler.hpp>

#include <imgui/imgui.h>
#include <ImGuizmo.h>

namespace stm2 {

Inspector::Inspector(Node& node) : mNode(node) {
	setInspectCallback<Device>();
	setInspectCallback<Instance>();
	setInspectCallback<Swapchain>();
	setInspectCallback<Window>();
}

// scene graph node inspector
bool Inspector::drawNodeGui(Node& n) {
	ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnDoubleClick | ImGuiTreeNodeFlags_OpenOnArrow;
	if (&n == mSelected.get()) flags |= ImGuiTreeNodeFlags_Selected;
	if (n.children().empty()) flags |= ImGuiTreeNodeFlags_Leaf;

	// open nodes above selected node
	if (mSelected && n.isDescendant(*mSelected))
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
		const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("SceneNode");
		if (payload) {
			Node* nodeptr = *(Node**)payload->Data;
			if (nodeptr) {
				n.addChild(nodeptr->getPtr());
				if (auto s = n.findAncestor<Scene>())
					s->markDirty();
			}
		}
		ImGui::EndDragDropTarget();
	}

	bool erase = false;

	// context menu
	if (ImGui::BeginPopupContextItem()) {
		if (ImGui::Button("Delete")) {
			erase = true;
		}

		if (ImGui::Button("Add child")) {
			ImGui::OpenPopup("Add node");
		}

		if (ImGui::Button("Add component")) {
			ImGui::OpenPopup("Add component");
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

		// add component dialog
		if (ImGui::BeginPopup("Add component")) {
			for (auto[type, ctor] : mInspectorConstructFns) {
				if (n.hasComponent(type))
					continue;

				if (ImGui::Button(type.name())) {
					n.addComponent(type, ctor(n));
					ImGui::CloseCurrentPopup();
				}
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
			if (mSelected.get() == c) {
				select(nullptr);
			}
		}

		ImGui::TreePop();
	}
	return erase;
}

void Inspector::draw() {
	ProfilerScope ps("Inspector::draw");

	if (ImGui::Begin("Node Graph")) {
		drawNodeGui(*mNode.root());
	}
	ImGui::End();

	if (ImGui::Begin("Inspector")) {
		if (mSelected) {
			auto s = mSelected->findAncestor<Scene>();
			ImGui::PushID(s.get());
			if ((s && !mSelected->getComponent<Scene>()) && ImGui::Button("x")) {
				// delete selected node

				if (const shared_ptr<Node> p = mSelected->parent(); p && ImGui::GetIO().KeyAlt) {
					for (const shared_ptr<Node>& c : mSelected->children())
						p->addChild(c);
				}
				// detach from tree
				mSelected->removeParent();
				select(nullptr);
				s->markDirty();
			} else {
				// inspect selected node

				ImGui::SameLine();
				ImGui::Text("%s", mSelected->name().c_str());
				ImGui::SetNextItemWidth(40);

				// list components in selected node

				type_index to_erase = typeid(nullptr_t);
				for (type_index type : mSelected->components()) {
					if ((s && !mSelected->getComponent<Scene>())) {
						ImGui::PushID(type.hash_code());
						if (ImGui::Button("x")) to_erase = type;
						ImGui::PopID();
						ImGui::SameLine();
					}
					if (ImGui::CollapsingHeader(type.name(), ImGuiTreeNodeFlags_DefaultOpen)) {
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