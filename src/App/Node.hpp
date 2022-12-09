#pragma once

#include <memory>
#include <vector>
#include <stack>
#include <unordered_map>
#include <typeindex>
#include <ranges>

namespace stm2 {

using namespace std;

class Node;

// Scene graph node. Nodes hold pointers to parent/children nodes, as well as Components (just shared_ptr's of arbitrary types).
class Node : public enable_shared_from_this<Node> {
private:
	string mName;
	unordered_map<type_index, shared_ptr<void>> mComponents;

	weak_ptr<Node> mParent;
	vector<shared_ptr<Node>> mChildren;

	Node(const string& name) : mName(name) {}

public:
	[[nodiscard]] inline static shared_ptr<Node> create(const string& name) {
		return shared_ptr<Node>(new Node(name));
	}

	Node(Node&&) = default;
	Node& operator=(Node&&) = default;

	Node(const Node&) = delete;
	Node& operator=(const Node&) = delete;

	inline const string& name() const { return mName; }
	inline shared_ptr<Node> getPtr() { return shared_from_this(); }

	// Components

	inline auto components() const { return mComponents | views::keys; }

	template<typename T>
	inline void addComponent(const shared_ptr<T>& v) {
		mComponents.emplace(typeid(T), v);
	}

	template<typename T>
	inline void removeComponent() {
		auto it = mComponents.find(typeid(T));
		if (it != mComponents.end()) {
			it->second.reset();
			mComponents.erase(it);
		}
	}
	inline void removeComponent(const type_index type) {
		mComponents.erase(type);
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
		return static_pointer_cast<T>(it->second);
	}

	// Relations

	inline shared_ptr<Node> parent() const { return mParent.lock(); }
	inline const auto& children() const { return mChildren; }

	inline shared_ptr<Node> root() {
		shared_ptr<Node> n = shared_from_this();
		while (shared_ptr<Node> p = n->parent()) {
			n = p;
		}
		return n;
	}

	inline void addChild(const shared_ptr<Node>& c) {
		if (!c) return;
		c->removeParent();
		c->mParent = shared_from_this();
		mChildren.emplace_back(c);
	}
	inline shared_ptr<Node> addChild(const string& name) {
		const shared_ptr<Node> c = create(name);
		addChild(c);
		return c;
	}
	inline void removeChild(Node& c) {
		for (auto it = mChildren.begin(); it != mChildren.end(); it++) {
			if (it->get() == &c) {
				c.mParent.reset();
				mChildren.erase(it);
				break;
			}
		}
	}
	inline void removeParent() {
		if (auto p = parent())
			p->removeChild(*this);
	}

	// forEach

	template<invocable<Node&> F>
	inline void forEachDescendant(F&& fn) {
		stack<shared_ptr<Node>> todo;
		todo.push(shared_from_this());
		while (!todo.empty()) {
			const shared_ptr<Node> n = todo.top();
			todo.pop();
			fn(*n);
			for (const shared_ptr<Node>& c : n->mChildren)
				todo.push(c);
		}
	}
	template<invocable<Node&> F>
	inline void forEachAncestor(F&& fn) {
		shared_ptr<Node> n = shared_from_this();
		while (n) {
			fn(*n);
			n = n->parent();
		}
	}

	// forEach component

	template<typename T, invocable<Node&, shared_ptr<T>> F>
	inline void forEachDescendant(F&& fn) {
		forEachDescendant([&](Node& n) {
			if (const shared_ptr<T> c = n.getComponent<T>(); c)
				fn(n, c);
		});
	}

	template<typename T, invocable<Node&, shared_ptr<T>> F>
	inline void forEachAncestor(F&& fn) {
		forEachAncestor([&](Node& n) {
			if (const shared_ptr<T> c = n.getComponent<T>(); c)
				fn(n, c);
		});
	}

	// find (stops when fn evaluates to false)

	template<typename F> requires(is_invocable_r_v<bool, F, Node&>)
	inline void findDescendant(F&& fn) {
		stack<shared_ptr<Node>> todo;
		todo.push(shared_from_this());
		while (!todo.empty()) {
			const shared_ptr<Node> n = todo.top();
			todo.pop();
			if (!fn(*n)) break;
			for (const shared_ptr<Node>& c : n->mChildren)
				todo.push(c);
		}
	}
	template<typename F> requires(is_invocable_r_v<bool, F, Node&>)
	inline void findAncestor(F&& fn) {
		shared_ptr<Node> n = shared_from_this();
		while (n) {
			if (!fn(*n)) break;
			n = n->parent();
		}
	}

	// returns true if p is an ancestor of this node
	inline bool isAncestor(Node& p) {
		bool found = false;
		findAncestor([&](Node& n){
			if (&n == &p) {
				found = true;
				return false;
			}
			return true;
		});
		return found;
	}
	// returns true if p is a descendant of this node
	inline bool isDescendant(Node& p) {
		bool found = false;
		findDescendant([&](Node& n){
			if (&n == &p) {
				found = true;
				return false;
			}
			return true;
		});
		return found;
	}

	// component find functions

	template<typename T>
	inline shared_ptr<T> findDescendant() {
		shared_ptr<T> ptr;
		findDescendant([&](Node& n) {
			if (const shared_ptr<T> c = n.getComponent<T>(); c) {
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
		findAncestor([&](Node& n) {
			if (const shared_ptr<T> c = n.getComponent<T>(); c) {
				ptr = c;
				return false;
			}
			return true;
		});
		return ptr;
	}
};

}