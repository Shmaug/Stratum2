#pragma once

#include "Device.hpp"

namespace tinyvkpt {

template<derived_from<Device::Resource> T>
class ResourcePool {
private:
	stack<shared_ptr<T>> mAvailable;
	list<shared_ptr<T>> mInFlight;
public:
	inline void clear() {
		mInFlight.clear();
		mAvailable = {};
	}

	inline size_t size() const {
		return mAvailable.size() + mInFlight.size();
	}

	inline shared_ptr<T> get() {
		shared_ptr<T> r;

		// return available resource, if any
		if (!mAvailable.empty()) {
			r = mAvailable.top();
			mAvailable.pop();
			return r;
		}

		for (auto it = mInFlight.begin(); it != mInFlight.end(); ) {
			// skip in-flight resources
			if ((*it)->inFlight()) {
				it++;
				continue;
			}

			// return the first available resource, store other available resources in mAvailable
			if (!r)
				r = *it; // returned resource stays in-flight
			else
				mAvailable.push(*it);
			it = mInFlight.erase(it);
		}

		return r;
	}

	inline shared_ptr<T>& emplace(const shared_ptr<T>& v) {
		return mInFlight.emplace_back(v);
	}
};

}