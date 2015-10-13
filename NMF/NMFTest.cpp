#include "stdafx.h"
#include "CppUnitTest.h"

#include "NMF.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace std;
namespace NMF
{		
	TEST_CLASS(NMFTest)
	{
		const int n = 100;
		const int m = 200;
		const int r = 10;
		Mat X;
		Mat Uinit;
		Mat Vinit;
	public:
		NMFTest() {
			srand(0);
			X = Mat::Random(n, m).cwiseAbs();
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

		TEST_METHOD(TestNMF_HALS)
		{
			Mat U = Uinit, V = Vinit;
			StandardProgressReporter progressReporter;
			NMF_HALS(X, r, U, V, progressReporter, DefaultConvergenceTester(100, -1));

			string outfile = string(__func__) +  ".log";
			WriteProgress(progressReporter, outfile);
		}
	};
}