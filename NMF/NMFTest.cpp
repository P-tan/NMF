#include "stdafx.h"
#include "CppUnitTest.h"

#include "NMF.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace std;
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

		TEST_METHOD(TestStandardProgressReporter)
		{
			const Mat X = Mat(10, 20).setRandom().cwiseAbs();
			const int r = 3;
			Mat U, V;
			StandardProgressReporter progressReporter;
			NMF_impl(X, r, U, V, 
				RandomInitializer(),
				NullUpdater(),
				DefaultConvergenceTester(100, -1), 
				progressReporter
				);
			Assert::AreEqual(101, (int)progressReporter.GetProgress().size());
			for (auto &prg : progressReporter.GetProgress())
			{
				cout <<
					prg.loop_no << ", " <<
					prg.l2norm << ", " <<
					prg.time << endl;
			}
		}
	};
}