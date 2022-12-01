#pragma once

#include <memory>
#include <deque>
#include <list>
#include <vector>
#include <stack>
#include <string>
#include <chrono>

#include <Utils/fwd.hpp>
#include <Utils/hlslmath.hpp>

namespace stm2 {

class Profiler {
public:
	inline static void beginSample(const string& label, const float4& color = float4::Ones()) {
		auto s = make_shared<ProfilerSample>(mCurrentSample, label, color);
		if (mCurrentSample)
			mCurrentSample = mCurrentSample->mChildren.emplace_back(s);
		else
			mCurrentSample = s;
	}
	inline static void endSample() {
		if (!mCurrentSample) throw logic_error("cannot call end_sample without first calling begin_sample");
		mCurrentSample->mDuration += chrono::high_resolution_clock::now() - mCurrentSample->mStartTime;
		if (!mCurrentSample->mParent && mSampleHistory.size() < mSampleHistoryCount)
			mSampleHistory.emplace_back(mCurrentSample);
		mCurrentSample = mCurrentSample->mParent;
	}

	inline static void setTimestamps(const chrono::steady_clock::time_point& t0, const vector<pair<string,chrono::nanoseconds>>& gpuTimestamps) {
		if (gpuTimestamps.size() && mTimestamps.size() < mSampleHistoryCount)
			mTimestamps.emplace_back(t0, gpuTimestamps);
	}
	inline static void beginFrame() {
		auto rn = chrono::high_resolution_clock::now();
		if (mFrameStart && mFrameTimeCount > 0) {
			auto duration = rn - *mFrameStart;
			mFrameTimes.emplace_back(chrono::duration_cast<chrono::duration<float, milli>>(duration).count());
			while (mFrameTimes.size() > mFrameTimeCount) mFrameTimes.pop_front();
		}
		mFrameStart = rn;
	}

	inline static bool hasHistory() { return !mSampleHistory.empty(); }
	inline static void resetHistory(uint32_t n) {
		mSampleHistoryCount = n;
		mSampleHistory.clear();
		mTimestamps.clear();
	}

	static void frameTimesGui();
	static void sampleTimelineGui();
	static void gpuTimestampGui();

	struct ProfilerSample {
		shared_ptr<ProfilerSample> mParent;
		list<shared_ptr<ProfilerSample>> mChildren;
		chrono::high_resolution_clock::time_point mStartTime;
		chrono::nanoseconds mDuration;
		float4 mColor;
		string mLabel;

		ProfilerSample() = default;
		ProfilerSample(const ProfilerSample& s) = default;
		ProfilerSample(ProfilerSample&& s) = default;
		inline ProfilerSample(const shared_ptr<ProfilerSample>& parent, const string& label, const float4& color)
			: mParent(parent), mColor(color), mLabel(label), mStartTime(chrono::high_resolution_clock::now()), mDuration(chrono::nanoseconds::zero()) {}
	};

private:
	static shared_ptr<ProfilerSample> mCurrentSample;
	static vector<pair<chrono::steady_clock::time_point, vector<pair<string,chrono::nanoseconds>>>> mTimestamps;
	static vector<shared_ptr<ProfilerSample>> mSampleHistory;
	static uint32_t mSampleHistoryCount;
	static optional<chrono::high_resolution_clock::time_point> mFrameStart;
	static deque<float> mFrameTimes;
	static uint32_t mFrameTimeCount;
};

class ProfilerScope {
private:
	const CommandBuffer* mCommandBuffer;

public:
	ProfilerScope(const string& label, const CommandBuffer* cmd = nullptr, const float4& color = float4::Ones());
	~ProfilerScope();
};

}