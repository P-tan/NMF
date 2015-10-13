#include "stdafx.h"
#include "CppUnitTest.h"

#include "NMF.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace std;
namespace NMF
{		
	TEST_CLASS(NMFTest)
	{
#ifndef NDEBUG
		const int m = 10;
		const int n = 20;
		const int r = 3;
#else 
		const int m = 100;
		const int n = 200;
		const int r = 30;
#endif
		Mat X;
		Mat Uinit;
		Mat Vinit;
	public:
		NMFTest() {
			srand(0);
			X = Mat::Random(m, n).cwiseAbs();
			RandomInitializer()(X, r, Uinit, Vinit);
		}
		void WriteProgress(
			const StandardProgressReporter &progressReporter, 
			const string &outfile
			) const 
		{
			ofstream ofs(outfile);
			Assert::IsTrue(ofs.is_open());
			ofs << StandardProgressReporter::Progress::Header() << endl;
			for (auto &prg : progressReporter.GetProgress())
			{
				prg.DebugPrint(ofs) << endl;
				Assert::IsFalse(!ofs);
			}
		}

		void WriteMat(
			const Mat &m,
			const string &outfile
			)
		{
			ofstream ofs(outfile);
			Assert::IsTrue(ofs.is_open());
			ofs << m; 
		}
		
		TEST_METHOD(TestNMF_impl)
		{
			Mat U = Uinit, V = Vinit;
			NMF_impl(X, r, U, V);
		}

		TEST_METHOD(TestStandardProgressReporter)
		{
			Mat U = Uinit, V = Vinit;
			StandardProgressReporter progressReporter;
			NMF_impl(X, r, U, V, 
				progressReporter,
				DefaultConvergenceTester(100, -1), 
				NullUpdater()
				);
			Assert::AreEqual(101, (int)progressReporter.GetProgress().size());

			string outfile = string(__func__) +  ".log";
			WriteProgress(progressReporter, outfile);
		}

		TEST_METHOD(TestNMF_MU)
		{
			Mat U = Uinit, V = Vinit;
			StandardProgressReporter progressReporter;
			NMF_MU(X, r, U, V, progressReporter, DefaultConvergenceTester(100, -1));

			string outfile = string(__func__) +  ".log";
			WriteProgress(progressReporter, outfile);
		}

		TEST_METHOD(TestNMF_FastHALS)
		{
			Mat U = Uinit, V = Vinit;
			StandardProgressReporter progressReporter;
			NMF_FastHALS(X, r, U, V, progressReporter, DefaultConvergenceTester(100, -1));

			string outfile = string(__func__) +  ".log";
			WriteProgress(progressReporter, outfile);
		}
	};
}