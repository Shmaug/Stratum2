#pragma once

#include "Node.hpp"

#include <functional>
#include <variant>

namespace stm2 {

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

	inline const shared_ptr<Node>& selected() const { return mSelected; }
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

	template<typename T> requires(has_drawGui_node<T> || has_drawGui_void<T>)
	inline void setTypeCallback() {
		if (mInspectorGuiFns.find(typeid(T)) == mInspectorGuiFns.end())
			mInspectorGuiFns[typeid(T)] = [](Node& n) {
				if constexpr(has_drawGui_node<T>)
					n.getComponent<T>()->drawGui(n);
				else
					n.getComponent<T>()->drawGui();
			};
	}
	template<typename T, invocable<Node&> F>
	inline void setTypeCallback(F&& fn) {
		if (mInspectorGuiFns.find(typeid(T)) == mInspectorGuiFns.end())
			mInspectorGuiFns[typeid(T)] = fn;
	}

	template<typename T>
	inline void unsetTypeCallback() {
		mInspectorGuiFns.erase(typeid(T));
	}
	inline void unsetTypeCallback(type_index t) {
		mInspectorGuiFns.erase(t);
	}

private:
	unordered_map<type_index, function<void(Node&)>> mInspectorGuiFns;
	shared_ptr<Node> mSelected;
	unordered_map<Node*, PinnedComponent> mPinned;

	void drawNodeGui(Node& n);
};

}