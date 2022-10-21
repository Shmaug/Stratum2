#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include <typeindex>

namespace tinyvkpt {

using namespace std;

class Node {
private:
	struct Component {};
	template<typename T>
	struct ComponentWrapper : public Component { shared_ptr<T> mObject; };

	weak_ptr<Node> mParent;
	vector<shared_ptr<Node>> mChildren;
	unordered_map<type_index, shared_ptr<Component>> mComponents;

public:
	template<typename T>
	inline T& addComponent(T&& v) {
		mComponents.emplace(typeid(T), make_shared<ComponentWrapper<T>>(move(v)));
	}
	template<typename T>
	inline T& addComponent(const T& v) {
		mComponents.emplace(typeid(T), make_shared<ComponentWrapper<T>>(v));
	}
	template<typename T, typename...Args>
	inline T& addComponent(Args&&... v) {
		mComponents.emplace(typeid(T), make_shared<ComponentWrapper<T>>(forward(v)...));
	}

	template<typename T>
	inline shared_ptr<T> getComponent() const {
		auto it = mComponents.find(typeid(T));
		if (it == mComponents.end())
			return nullptr;
		return static_pointer_cast<ComponentWrapper<T>>(it->second);
	}

	inline shared_ptr<Node> parent() const {
		return mParent.lock();
	}
};

}