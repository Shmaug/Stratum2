#pragma once

#include "Device.hpp"

namespace tinyvkpt {

template<derived_from<Device::Resource> T>
class DeviceResourcePool {
private:
	list<shared_ptr<T>> mResources;

public:
	inline void clear() {
		mResources.clear();
	}

	inline size_t size() const {
		return mResources.size();
	}

	inline shared_ptr<T> get() {
		for (const shared_ptr<T>& r : mResources)
			if (!r->inFlight())
				return r;
		return nullptr;
	}

	inline shared_ptr<T>& emplace(const shared_ptr<T>& v) {
		return mResources.emplace_back(v);
	}
};

}