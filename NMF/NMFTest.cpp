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

			string outfile = string(__func__) +  ".log";
			ofstream ofs(outfile);
			Assert::IsTrue(ofs.is_open());
			ofs << StandardProgressReporter::Progress::Header() << endl;
			for (auto &prg : progressReporter.GetProgress())
			{
				prg.DebugPrint(ofs) << endl;
				Assert::IsFalse(!ofs);
			}
		}
	};
}