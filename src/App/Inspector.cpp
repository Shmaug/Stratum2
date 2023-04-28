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

// dont allow deletion of core components
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
		if (ImGui::Selectable("Add component")) {
			ImGui::OpenPopup("Add component");
		}
		if (ImGui::Selectable("Add child")) {
			ImGui::OpenPopup("Add node");
		}
		bool canDelete = true;
		for (type_index c : n.components()) {
			if (gProtectedComponents.contains(c)) {
				canDelete = false;
				break;
			}
		}
		if (canDelete && ImGui::Selectable("Delete")) {
			erase = true;
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
		drawNodeGui(*mNode.root());
	}
	ImGui::End();

	if (ImGui::Begin("Inspector")) {
		if (auto sel = selected()) {
			ImGui::PushID(sel.get());
			bool deleteNode = ImGui::Button("x");

			{
				// inspect selected node

				ImGui::SameLine();
				ImGui::Text("%s", sel->name().c_str());
				ImGui::SetNextItemWidth(40);

				// inspect components in node

				type_index to_erase = typeid(nullptr_t);
				for (type_index type : sel->components()) {
					if (gProtectedComponents.contains(type)) {
						deleteNode = false;
					} else {
						ImGui::PushID(type.hash_code());
						if (ImGui::Button("x"))
							to_erase = type;
						ImGui::PopID();
						ImGui::SameLine();
					}

					if (ImGui::CollapsingHeader(type.name(), ImGuiTreeNodeFlags_DefaultOpen)) {
						if (auto it = mInspectorGuiFns.find(type); it != mInspectorGuiFns.end()) {
							// draw gui for component
							ImGui::Indent();
							ImGui::Indent();
							it->second(*sel);
							ImGui::Unindent();
							ImGui::Unindent();
						}
					}
				}
				if (to_erase != typeid(nullptr_t)) {
					sel->removeComponent(to_erase);
					if (auto scene = sel->findAncestor<Scene>())
						scene->markDirty();
				}
			}

			if (deleteNode) {
				if (const shared_ptr<Node> p = sel->parent(); p && ImGui::GetIO().KeyAlt) {
					for (const shared_ptr<Node>& c : sel->children())
						p->addChild(c);
				}
				// detach from tree
				sel->removeParent();
				auto scene = sel->findAncestor<Scene>();
				select(nullptr);
				if (scene)
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