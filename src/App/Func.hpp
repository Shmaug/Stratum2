#pragma once

// Slang function ptr (slang/source/core/slang-func-ptr.h)

namespace stm2 {

template <typename TResult, typename... Arguments>
class FuncPtr {
public:
	virtual TResult operator()(Arguments...) const = 0;
	virtual bool operator==(const FuncPtr<TResult, Arguments...>*) { return false; }
	virtual FuncPtr<TResult, Arguments...>* clone() = 0;
	virtual ~FuncPtr() {}
};

template <typename TResult, typename... Arguments>
class CdeclFuncPtr : public FuncPtr<TResult, Arguments...> {
public:
	typedef TResult(*FuncType)(Arguments...);

private:
	FuncType funcPtr;

public:
	CdeclFuncPtr(FuncType func)
		: funcPtr(func) {
	}
	virtual FuncPtr<TResult, Arguments...>* clone() override {
		auto rs = new CdeclFuncPtr<TResult, Arguments...>(funcPtr);
		return rs;
	}

	virtual TResult operator()(Arguments... params) const override { return funcPtr(params...); }

	virtual bool operator==(const FuncPtr<TResult, Arguments...>* ptr) override {
		auto cptr = dynamic_cast<const CdeclFuncPtr<TResult, Arguments...>*>(ptr);
		if (cptr)
			return funcPtr == cptr->funcPtr;
		else
			return false;
	}
};

template <typename Class, typename TResult, typename... Arguments>
class MemberFuncPtr : public FuncPtr<TResult, Arguments...> {
public:
	typedef TResult(Class::* FuncType)(Arguments...);

private:
	FuncType funcPtr;
	Class* object;

public:
	MemberFuncPtr(Class* obj, FuncType func)
		: funcPtr(func)
		, object(obj) {
	}
	virtual FuncPtr<TResult, Arguments...>* clone() override {
		auto rs = new MemberFuncPtr<Class, TResult, Arguments...>(object, funcPtr);
		return rs;
	}
	virtual TResult operator()(Arguments... params) const override {
		return (object->*funcPtr)(params...);
	}

	virtual bool operator==(const FuncPtr<TResult, Arguments...>* ptr) override {
		auto cptr = dynamic_cast<const MemberFuncPtr<Class, TResult, Arguments...>*>(ptr);
		if (cptr)
			return funcPtr == cptr->funcPtr && object == cptr->object;
		else
			return false;
	}
};

template <typename F, typename TResult, typename... Arguments>
class LambdaFuncPtr : public FuncPtr<TResult, Arguments...> {
private:
	F func;

public:
	LambdaFuncPtr(const F& _func)
		: func(_func) {
	}
	virtual TResult operator()(Arguments... params) const override { return func(params...); }
	virtual FuncPtr<TResult, Arguments...>* clone() override {
		auto rs = new LambdaFuncPtr<F, TResult, Arguments...>(func);
		return rs;
	}
	virtual bool operator==(const FuncPtr<TResult, Arguments...>* /*ptr*/) override {
		return false;
	}
};

template <typename TResult, typename... Arguments>
class Func {
private:
	shared_ptr<FuncPtr<TResult, Arguments...>> funcPtr;

public:
	Func() {}
	Func(typename CdeclFuncPtr<TResult, Arguments...>::FuncType func) {
		funcPtr = make_shared<CdeclFuncPtr<TResult, Arguments...>>(func);
	}
	template <typename Class>
	Func(Class* object, typename MemberFuncPtr<Class, TResult, Arguments...>::FuncType func) {
		funcPtr = make_shared<MemberFuncPtr<Class, TResult, Arguments...>>(object, func);
	}
	template <typename TFuncObj> Func(const TFuncObj& func) {
		funcPtr = make_shared<LambdaFuncPtr<TFuncObj, TResult, Arguments...>>(func);
	}
	Func& operator=(typename CdeclFuncPtr<TResult, Arguments...>::FuncType func) {
		funcPtr = make_shared<CdeclFuncPtr<TResult, Arguments...>>(func);
		return *this;
	}
	template <typename Class>
	Func& operator=(const MemberFuncPtr<Class, TResult, Arguments...>& func) {
		funcPtr = make_shared<MemberFuncPtr<Class, TResult, Arguments...>>(func);
		return *this;
	}
	template <typename TFuncObj> Func& operator=(const TFuncObj& func) {
		funcPtr = make_shared<LambdaFuncPtr<TFuncObj, TResult, Arguments...>>(func);
		return *this;
	}
	bool operator==(const Func& f) { return *funcPtr == f.funcPtr.Ptr(); }
	bool operator!=(const Func& f) { return !(*this == f); }
	operator bool() const { return funcPtr != nullptr; }
	TResult operator()(Arguments... params) const { return (*funcPtr)(params...); }
};

}