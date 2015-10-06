#include "stdafx.h"
#include "CppUnitTest.h"

#include "NMF.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace NMF
{		
	TEST_CLASS(NMFTest)
	{
	public:
		
		TEST_METHOD(TestNMF)
		{
			const Mat X = Mat(10, 20).setRandom().cwiseAbs();
			const int r = 3;
			Mat U, V;
			NMF_impl(X, r, U, V);
		}

	};
}