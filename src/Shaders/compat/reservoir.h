#pragma once

#include "hlslcompat.h"

STM_NAMESPACE_BEGIN

struct Reservoir {
	float mTotalWeight;
    float mSampleTargetPdf;
    uint mCandidateCount;

    SLANG_CTOR(Reservoir) () {
        mTotalWeight = 0;
        mSampleTargetPdf = 0;
        mCandidateCount = 0;
	}

	SLANG_MUTATING
    inline bool update(const float rnd, const float sourcePdf, const float targetPdf) {
		const float w = targetPdf / sourcePdf;
        mTotalWeight += w;
		mCandidateCount++;
        if (rnd * mTotalWeight <= w) {
            mSampleTargetPdf = targetPdf;
            return true;
        }
        return false;
	}

    inline float W() CONST_CPP {
        return mTotalWeight / (mCandidateCount * mSampleTargetPdf);
	}
};

STM_NAMESPACE_END