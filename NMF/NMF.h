#pragma once
#include <cassert>
#include <iosfwd>
#include <string>
#include <vector>
#include <boost/timer.hpp>
#include <Eigen/Core>

typedef Eigen::MatrixXd Mat;
typedef Eigen::VectorXd Vec;

//
// Initializer
//

class RandomInitializer
{
public:
	void operator()(
		const Mat &X,
		int r,
		Mat &U,
		Mat &V
		)
	{
		U = Mat::Random(X.rows(), r).cwiseAbs();
		V = Mat::Random(r, X.cols()).cwiseAbs();
	}
};

//
// Updater 
//

class NullUpdater
{
public:
	void operator()(
		const Mat &X,
		Mat &U,
		Mat &V)
	{
	}
};

//! @brief Multicative Update 
//! @see Daniel D. Lee and H. Sebastian Seung (2001). "Algorithms for Non-negative Matrix Factorization". Advances in Neural Information Processing Systems 13: Proceedings of the 2000 Conference. MIT Press. pp. 556?562.
class MUUpdater
{
public:
	void operator()(
		const Mat &X,
		Mat &U,
		Mat &V)
	{
		assert(U.rows() == X.rows());
		assert(U.cols() == V.rows());
		assert(V.cols() == X.cols());
        U.array() = U.array() * (X * V.transpose()).array() / 
                     (U * V * V.transpose()).array();
        V.array() = V.array() * (U.transpose() * X).array() /
                     (U.transpose() * U * V).array();
	}
};

//! @brief Hierarchical Alternating Least Squares Algorithms
//! @see Cichocki, A., & Anh-Huy, P. (2009). Fast local algorithms for large scale nonnegative matrix and tensor factorizations. IEICE Transactions on Fundamentals of Electronics, Communications and Computer Sciences, 92(3), 708?721. http://www.bsp.brain.riken.jp/publications/2009/Cichocki-Phan-IEICE_col.pdf
class FastHALSUpdater
{
	const double eps = 1e-8;
	Mat A;
	Mat B;
public:
	void operator()(
		const Mat &X,
		Mat &U,
		Mat &V)
	{
		assert(U.rows() == X.rows());
		assert(U.cols() == V.rows());
		assert(V.cols() == X.cols());
		A = X * V.transpose();
		B = V * V.transpose();
		const int K = U.cols();
		for (int k = 0; k < K; ++k) {
			const Vec Ak = A.col(k);
			const Vec Bk = B.col(k);
			const Vec uk = U.col(k);
			U.col(k) = ((Ak - U * Bk + uk * B(k, k)) / B(k, k)).cwiseMax(eps);
		}
		A = X.transpose() * U;
		B = U.transpose() * U;
		for (int k = 0; k < K; ++k) {
			const Vec Ak = A.col(k);
			const Vec Bk = B.col(k);
			const Vec vk = V.row(k);
			V.row(k) = ((Ak - V.transpose() * Bk + vk * B(k, k)) / B(k, k)).cwiseMax(eps);
		}
	}
};

//! @brief Greedy Coordinate Descent Algorithm
//! @see Hsieh, C.-J., & Dhillon, I. S. (2011). Fast coordinate descent methods with variable selection for non-negative matrix factorization. In ACM SIGKDD (pp. 1064?1072)
class GCDUpdater
{
	Mat A;
	Mat B;
	Mat S;
	Mat D;
public:
	void operator()(
		const Mat &X,
		Mat &U,
		Mat &V)
	{
		B = V.transpose() * V;
		A = U * B - X * V.transpose();
		S = (U - A * B.cwiseInverse().diagonal()).cwiseMax(0) - U;
		D = -A.array() * S.array() - B.diagonal() * S.pow(2.0) / 2;
	}
};

//
// ConvergenceTester
//

class DefaultConvergenceTester
{
	int m_max_loop_count;
	double m_eps;
public:
	DefaultConvergenceTester(
		int max_loop_count = 100,
		double eps = 1e-7
		)
		: m_max_loop_count(max_loop_count)
		, m_eps(eps)
	{}

	bool operator()(
		const Mat &X,
		const Mat &U,
		const Mat &V,
		int loop_count
		) const
	{
		if (loop_count >= m_max_loop_count) {
			return true;
		}
		// Normalized Residual Value 
		const double nrv = (X - U * V).squaredNorm() / X.squaredNorm();
		if (nrv < m_eps)
		{
			return true;
		}
		return false;
	}
};

//
// Progress Reporter
//

class NullProgressReporter
{
public:
	void Initialize() {}
	void Report(
		const Mat &X,
		const Mat &U,
		const Mat &V,
		int loop_count
		)
	{}
};

class StandardProgressReporter
{
public:
	struct Progress {
		int loop_no;
		double nrv; //!< Normalized Residual Value
		double time;

		static const char * Header() { return "loop_no, NRV, Time_msec"; }
		std::ostream& DebugPrint(
			std::ostream& os
			) const
		{
			os <<
				loop_no << ", " <<
				nrv << ", " <<
				time * 1000;
			return os;
		}

	};
private:
	std::vector<Progress> m_progress;
	boost::timer m_timer;
public:
	StandardProgressReporter()
		: m_progress()
		, m_timer()
	{
	}

	void Initialize() {
		m_timer.restart();
	}

	void Report(
		const Mat &X,
		const Mat &U,
		const Mat &V,
		int loop_count
		)
	{
		const double nrv = (X - U * V).squaredNorm() / X.squaredNorm();
		Progress progress = {};
		progress.loop_no = loop_count;
		progress.nrv = nrv;
		progress.time = m_timer.elapsed();
		m_progress.push_back(progress);
	}

	const std::vector<Progress>& GetProgress() const { return m_progress; }
};

//
// NMF 
// 

//! @brief NMF Implementation 
//!
//! X = UV, X : n x m, U : n x r, V : r x m
template<
	class ProgressReporter = NullProgressReporter,
	class ConvergenceTester = DefaultConvergenceTester,
	class Updater = NullUpdater
>
void NMF_impl(
	const Mat &X,
	int r,
	Mat &U,
	Mat &V,
	ProgressReporter &progressReporter = ProgressReporter(),
	ConvergenceTester convergenceTester = ConvergenceTester(),
	Updater updater = Updater()
	)
{
	assert(X.minCoeff() >= 0);

	progressReporter.Initialize();

	int loop_count = 0;
	progressReporter.Report(X, U, V, loop_count);
	do {
		updater(X, U, V);
		++loop_count;
		progressReporter.Report(X, U, V, loop_count);
	} while (!convergenceTester(X, U, V, loop_count));
}

//! @brief NMF by multicative updating 
//!
//! X = UV, X : n x m, U : n x r, V : r x m
template<
	class ProgressReporter = NullProgressReporter,
	class ConvergenceTester = DefaultConvergenceTester
>
void NMF_MU(
	const Mat &X,
	int r,
	Mat &U,
	Mat &V,
	ProgressReporter &progressReporter = ProgressReporter(),
	ConvergenceTester convergenceTester = ConvergenceTester()
	)
{
	NMF_impl<ProgressReporter, ConvergenceTester, MUUpdater>(
		X, r, U, V, progressReporter, convergenceTester, MUUpdater());
}
//! @brief NMF by HALS
//!
//! X = UV, X : n x m, U : n x r, V : r x m
template<
	class ProgressReporter = NullProgressReporter,
	class ConvergenceTester = DefaultConvergenceTester
>
void NMF_FastHALS(
	const Mat &X,
	int r,
	Mat &U,
	Mat &V,
	ProgressReporter &progressReporter = ProgressReporter(),
	ConvergenceTester convergenceTester = ConvergenceTester()
	)
{
	NMF_impl<ProgressReporter, ConvergenceTester, FastHALSUpdater>(
		X, r, U, V, progressReporter, convergenceTester, FastHALSUpdater());
}
