#pragma once

#include <imgui/imgui.h>
#include "Func.hpp"

namespace stm2 {

class GuiProperty {
public:
	virtual bool drawGui() = 0;
};

template<typename T> requires(is_scalar_v<T>)
class PointerProperty : public GuiProperty {
public:
	string mLabel;
	T* mPointer;
	T mMin = 0;
	T mMax = 0;
	float mDragSpeed = 1;

	PointerProperty() = default;
	PointerProperty(const PointerProperty&) = default;
	PointerProperty(PointerProperty&&) = default;
	PointerProperty(const string& label, T* ptr, T vmin = 0, T vmax = 0, float vspeed = 1)
		: mLabel(label), mPointer(ptr), mMin(vmin), mMax(vmax), mDragSpeed(vspeed) {}

	static ImGuiDataType getType() {
		if constexpr(is_floating_point_v<T>)
			return sizeof(T) == sizeof(float) ? ImGuiDataType_Float : ImGuiDataType_Double;
		else if constexpr (sizeof(T) == sizeof(uint64_t))
			return is_signed_v<T> ? ImGuiDataType_S64 : ImGuiDataType_U64;
		else if constexpr (sizeof(T) == sizeof(uint32_t))
			return is_signed_v<T> ? ImGuiDataType_S32 : ImGuiDataType_U32;
		else if constexpr (sizeof(T) == sizeof(uint16_t))
			return is_signed_v<T> ? ImGuiDataType_S16 : ImGuiDataType_U16;
		else
			return ImGuiDataType_COUNT;
	}

	bool drawGui() {
		if (!mPointer)
			return false;
		ImGui::SetNextItemWidth(40);
		if (mDragSpeed == 0 && mMin != mMax)
			return ImGui::SliderScalar(mLabel.c_str(), getType(), mPointer, &mMin, &mMax);
		else
			return ImGui::DragScalar(mLabel.c_str(), getType(), mPointer, mDragSpeed, &mMin, &mMax);
	}
};

template<>
class PointerProperty<bool> : public GuiProperty {
public:
	string mLabel;
	bool* mPointer;

	PointerProperty() = default;
	PointerProperty(const PointerProperty&) = default;
	PointerProperty(PointerProperty&&) = default;
	PointerProperty(const string& label, bool* ptr) : mLabel(label), mPointer(ptr) {}

	bool drawGui() {
		if (!mPointer)
			return false;
		return ImGui::Checkbox(mLabel.c_str(), mPointer);
	}
};

template<typename T> requires(is_scalar_v<T>)
class AccessorProperty : public GuiProperty {
public:
	string mLabel;
	Func<T*> mAccessor;
	T mMin = 0;
	T mMax = 0;
	float mDragSpeed = 1;

	AccessorProperty() = default;
	AccessorProperty(const AccessorProperty&) = default;
	AccessorProperty(AccessorProperty&&) = default;
	AccessorProperty(const string& label, auto fn, T vmin = 0, T vmax = 0, float vspeed = 1) :
		mLabel(label), mAccessor(fn), mMin(vmin), mMax(vmax), mDragSpeed(vspeed) {}

	bool drawGui() {
		if (!mAccessor)
			return false;
		T* ptr = mAccessor();
		if (!ptr)
			return false;
		ImGui::SetNextItemWidth(50);
		if (mDragSpeed == 0 && mMin != mMax)
			return ImGui::SliderScalar(mLabel.c_str(), PointerProperty<T>::getType(), ptr, &mMin, &mMax);
		else
			return ImGui::DragScalar  (mLabel.c_str(), PointerProperty<T>::getType(), ptr, mDragSpeed, &mMin, &mMax);
	}
};

template<>
class AccessorProperty<bool> : public GuiProperty {
public:
	string mLabel;
	Func<bool*> mAccessor;

	AccessorProperty() = default;
	AccessorProperty(const AccessorProperty&) = default;
	AccessorProperty(AccessorProperty&&) = default;
	AccessorProperty(const string& label, auto fn) : mLabel(label), mAccessor(fn) {}

	bool drawGui() {
		if (!mAccessor)
			return false;
		bool* ptr = mAccessor();
		if (!ptr)
			return false;
		return ImGui::Checkbox(mLabel.c_str(), ptr);
	}
};


class SeparatorProperty : public GuiProperty {
public:
	bool drawGui() {
		ImGui::Separator();
		return false;
	}
};


// these allow variable initiailization + GUI setup in one line

template<typename T> requires(is_scalar_v<T>)
inline shared_ptr<GuiProperty> initialize_property(const string& label, const T& defaultValue, T& value, const T vmin = 0, const T vmax = 0, const float vspeed = 1) {
	value = defaultValue;
	return make_shared<PointerProperty<T>>(label, &value, vmin, vmax, vspeed);
}
template<>
inline shared_ptr<GuiProperty> initialize_property<bool>(const string& label, const bool& defaultValue, bool& value, const bool vmin, const bool vmax, const float vspeed) {
	value = defaultValue;
	return make_shared<PointerProperty<bool>>(label, &value);
}

template<typename T> requires(is_scalar_v<T>)
inline shared_ptr<GuiProperty> initialize_property(const string& label, const T& defaultValue, const Func<T*>& accessor, const T vmin = 0, const T vmax = 0, const float vspeed = 1) {
	*accessor() = defaultValue;
	return make_shared<AccessorProperty<T>>(label, accessor, vmin, vmax, vspeed);
}
template<>
inline shared_ptr<GuiProperty> initialize_property<bool>(const string& label, const bool& defaultValue, const Func<bool*>& accessor, const bool vmin, const bool vmax, const float vspeed) {
	*accessor() = defaultValue;
	return make_shared<AccessorProperty<bool>>(label, accessor);
}

inline shared_ptr<GuiProperty> initialize_separator() {
	return make_shared<SeparatorProperty>();
}

}