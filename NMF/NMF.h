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
		U.resize(X.rows(), r);
		V.resize(r, X.cols());

		U = U.setRandom().cwiseAbs();
		V = V.setRandom().cwiseAbs();
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
class HALSUpdater
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
		Mat E = X - U * V;
		const int K = U.cols();
		for (int k = 0; k < K; ++k) {
			Vec uk = U.col(k);
			Vec vk = V.row(k);
			const Mat Xk = E + uk * vk.transpose();
			U.col(k) = (Xk * vk / vk.dot(vk)).cwiseMax(0);
			V.row(k) = (Xk.transpose() * uk / uk.dot(uk)).cwiseMax(0);
			E = Xk - U.col(k) * V.row(k);
		}
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
		const double l2norm = (X - U * V).norm();
		if (l2norm < m_eps)
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
		double l2norm;
		double time;

		static const char * Header() { return "loop_no, L2Norm, Time_msec"; }
		std::ostream& DebugPrint(
			std::ostream& os
			) const
		{
			os <<
				loop_no << ", " <<
				l2norm << ", " <<
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
		const double l2norm = (X - U * V).norm();
		Progress progress = {};
		progress.loop_no = loop_count;
		progress.l2norm = l2norm;
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
	class Initializer = RandomInitializer,
	class Updater = NullUpdater
>
void NMF_impl(
	const Mat &X,
	int r,
	Mat &U,
	Mat &V,
	ProgressReporter &progressReporter = ProgressReporter(),
	ConvergenceTester convergenceTester = ConvergenceTester(),
	Initializer initializer = Initializer(),
	Updater updater = Updater()
	)
{
	assert(X.minCoeff() >= 0);

	progressReporter.Initialize();

	initializer(X, r, U, V);

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
	class ConvergenceTester = DefaultConvergenceTester,
	class Initializer = RandomInitializer
>
void NMF_MU(
	const Mat &X,
	int r,
	Mat &U,
	Mat &V,
	ProgressReporter &progressReporter = ProgressReporter(),
	ConvergenceTester convergenceTester = ConvergenceTester(),
	Initializer initializer = Initializer()
	)
{
	NMF_impl<ProgressReporter, ConvergenceTester, Initializer, MUUpdater>(
		X, r, U, V, progressReporter, convergenceTester, initializer, MUUpdater());
}
//! @brief NMF by HALS
//!
//! X = UV, X : n x m, U : n x r, V : r x m
template<
	class ProgressReporter = NullProgressReporter,
	class ConvergenceTester = DefaultConvergenceTester,
	class Initializer = RandomInitializer
>
void NMF_HALS(
	const Mat &X,
	int r,
	Mat &U,
	Mat &V,
	ProgressReporter &progressReporter = ProgressReporter(),
	ConvergenceTester convergenceTester = ConvergenceTester(),
	Initializer initializer = Initializer()
	)
{
	NMF_impl<ProgressReporter, ConvergenceTester, Initializer, HALSUpdater>(
		X, r, U, V, progressReporter, convergenceTester, initializer, HALSUpdater());
}
