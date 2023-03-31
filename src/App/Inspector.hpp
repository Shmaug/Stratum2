#pragma once

#include "Node.hpp"

#include <functional>
#include <variant>

namespace stm2 {

template<typename T>
concept is_node_constructible = requires(Node& n) { new T(n); };

template<typename T>
concept has_drawGui_node = requires(T t, Node& n) { t.drawGui(n); };
template<typename T>
concept has_drawGui_void = requires(T t) { t.drawGui(); };

class Inspector {
public:
	using ComponentInspectorFn = variant<function<void()>, function<void(Node&)>>;
	using PinnedComponent = tuple<string, weak_ptr<Node>, ComponentInspectorFn>;

    Node& mNode;

	Inspector(Node& node);

	void draw();

	inline shared_ptr<Node> selected() const { return mSelected.lock(); }
	inline void select(const shared_ptr<Node>& n) { mSelected = n; }

	template<typename T> requires(has_drawGui_node<T> || has_drawGui_void<T>)
	inline void pin(Node& n, const shared_ptr<T>& c) {
		if (auto it = mPinned.find(&n); it != mPinned.end())
			mPinned.erase(it);
		else {
			if constexpr(has_drawGui_node<T>)
				mPinned.emplace(&n, PinnedComponent{ n.name() + "/" + typeid(T).name(), n.getPtr(), function<void(Node&)>(bind(&T::drawGui, c.get(), std::placeholders::_1)) });
			else
				mPinned.emplace(&n, PinnedComponent{ n.name() + "/" + typeid(T).name(), n.getPtr(), function<void()>(bind(&T::drawGui, c.get())) });
		}
	}
	inline void unpin(Node& n) {
		mPinned.erase(&n);
	}

	template<typename T, invocable<Node&> F>
	inline void setInspectCallback(F&& fn) {
		if (mInspectorGuiFns.find(typeid(T)) == mInspectorGuiFns.end())
			mInspectorGuiFns[typeid(T)] = fn;
	}
	template<typename T> requires(has_drawGui_node<T> || has_drawGui_void<T>)
	inline void setInspectCallback() {
		setInspectCallback<T>([](Node& n) {
			if constexpr(has_drawGui_node<T>)
				n.getComponent<T>()->drawGui(n);
			else
				n.getComponent<T>()->drawGui();
		});
		if constexpr(is_default_constructible_v<T>)
			mInspectorConstructFns[typeid(T)] = [](Node& n){ return static_pointer_cast<void>(make_shared<T>()); };
		else if constexpr(is_node_constructible<T>)
			mInspectorConstructFns[typeid(T)] = [](Node& n){ return static_pointer_cast<void>(make_shared<T>(n)); };
	}

private:
	unordered_map<type_index, function<void(Node&)>> mInspectorGuiFns;
	unordered_map<type_index, function<shared_ptr<void>(Node& n)>> mInspectorConstructFns;
	unordered_map<Node*, PinnedComponent> mPinned;
	weak_ptr<Node> mSelected;
	string mInputChildName;

	bool drawNodeGui(Node& n);
};

}