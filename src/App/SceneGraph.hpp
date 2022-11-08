#pragma once

#include <memory>
#include <vector>
#include <stack>
#include <unordered_map>
#include <typeindex>

namespace tinyvkpt {

using namespace std;

class Node;

using NodePtr = shared_ptr<Node>;

class Node : enable_shared_from_this<Node> {
private:
	struct Component {};
	template<typename T>
	struct ComponentWrapper : public Component {
		shared_ptr<T> mComponent;
	};

	string mName;
	unordered_map<type_index, unique_ptr<Component>> mComponents;

	weak_ptr<Node> mParent;
	vector<NodePtr> mChildren;

	Node(const string& name) : mName(name) {}

public:
	[[nodiscard]] inline static NodePtr create(const string& name) {
		return NodePtr(new Node(name));
	}

	Node(Node&&) = default;
	Node& operator=(Node&&) = default;

	Node(const Node&) = delete;
	Node& operator=(const Node&) = delete;

	inline const string& name() const { return mName; }

	template<typename T>
	inline void addComponent(const shared_ptr<T>& v) {
		ComponentWrapper<T> w;
		w.mComponent = v;
		mComponents.emplace(typeid(T), make_unique<ComponentWrapper<T>>(w));
	}

	template<typename T, typename...Types>
	inline shared_ptr<T> makeComponent(Types&&... args) {
		shared_ptr<T> c = make_shared<T>( forward<Types>(args)... );
		addComponent(c);
		return c;
	}

	template<typename T>
	inline shared_ptr<T> getComponent() const {
		auto it = mComponents.find(typeid(T));
		if (it == mComponents.end())
			return nullptr;
		return static_cast<ComponentWrapper<T>*>(it->second.get())->mComponent;
	}

	inline NodePtr parent() const {
		return mParent.lock();
	}

	inline void addChild(const NodePtr& c) {
		if (!c) return;
		c->mParent = shared_from_this();
		mChildren.emplace_back(c);
	}
	inline void removeChild(Node& c) {
		c.mParent.reset();
		for (auto it = mChildren.begin(); it != mChildren.end(); it++) {
			if (it->get() == &c) {
				mChildren.erase(it);
				break;
			}
		}
	}
	inline void removeParent() {
		if (auto p = parent(); p)
			p->removeChild(*this);
	}

	inline NodePtr makeChild(const string& name) {
		const NodePtr c = create(name);
		addChild(c);
		return c;
	}

	// forEach

	template<invocable<NodePtr> F>
	inline void forEachDescendant(F&& fn) {
		stack<NodePtr> todo;
		todo.push(shared_from_this());
		while (!todo.empty()) {
			const NodePtr n = todo.top();
			todo.pop();
			fn(n);
			for (const NodePtr& c : n->mChildren)
				todo.push(c);
		}
	}
	template<invocable<NodePtr> F>
	inline void forEachAncestor(F&& fn) {
		NodePtr n = shared_from_this();
		while (n) {
			fn(n);
			n = n->parent();
		}
	}

	// forEach component

	template<typename T, invocable<NodePtr, shared_ptr<T>> F>
	inline void forEachDescendant(F&& fn) {
		forEachDescendant([&](const NodePtr& n) {
			if (auto c = getComponent<T>())
				fn(n, c);
		});
	}

	template<typename T, invocable<NodePtr, shared_ptr<T>> F>
	inline void forEachAncestor(F&& fn) {
		forEachAncestor([&](const NodePtr& n) {
			if (auto c = getComponent<T>())
				fn(n, c);
		});
	}

	// find (stops when fn evaluates to false)

	template<invocable<NodePtr> F>
	inline void findDescendant(F&& fn) {
		stack<NodePtr> todo;
		todo.push(shared_from_this());
		while (!todo.empty()) {
			const NodePtr n = todo.top();
			todo.pop();
			if (!fn(n)) break;
			for (const NodePtr& c : n->mChildren)
				todo.push(c);
		}
	}
	template<invocable<NodePtr> F>
	inline void findAncestor(F&& fn) {
		NodePtr n = shared_from_this();
		while (n) {
			if (!fn(n)) break;
			n = n->parent();
		}
	}

	// component find functions

	template<typename T>
	inline shared_ptr<T> findDescendant() {
		shared_ptr<T> ptr;
		findDescendant([&](const NodePtr& n) {
			if (const shared_ptr<T> c = n->getComponent<T>(); c) {
				ptr = c;
				return false;
			}
			return true;
		});
		return ptr;
	}
	template<typename T>
	inline shared_ptr<T> findAncestor() {
		shared_ptr<T> ptr;
		findAncestor([&](const NodePtr& n) {
			if (const shared_ptr<T> c = n->getComponent<T>(); c) {
				ptr = c;
				return false;
			}
			return true;
		});
		return ptr;
	}
};

}