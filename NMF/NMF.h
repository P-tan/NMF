#pragma once
#include <cassert>
#include <vector>
#include <boost/timer.hpp>
#include <Eigen/Core>

typedef Eigen::MatrixXd Mat;

class RandomInitializer
{
public:
	void Initialize(
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

class NullUpdater
{
public:
	void Update(
		Mat &U,
		Mat &V)
	{
	}
};

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

	bool IsConverged(
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
//! @brief NMF Implementation 
//!
//! X = UV, X : n x m, U : n x r, V : r x m
template<
	class Initializer = RandomInitializer,
	class Updater = NullUpdater, 
	class ConvergenceTester = DefaultConvergenceTester,
	class ProgressReporter = NullProgressReporter
>
void NMF_impl(
	const Mat &X,
	int r,
	Mat &U,
	Mat &V,
	Initializer initializer = Initializer(),
	Updater updater = Updater(),
	ConvergenceTester convergenceTester = ConvergenceTester(),
	ProgressReporter &progressReporter = ProgressReporter()
	)
{
	assert(X.minCoeff() >= 0);

	progressReporter.Initialize();

	initializer.Initialize(X, r, U, V);

	int loop_count = 0;
	progressReporter.Report(X, U, V, loop_count);
	do {
		updater.Update(U, V);
		++loop_count;
		progressReporter.Report(X, U, V, loop_count);
	} while (!convergenceTester.IsConverged(X, U, V, loop_count));
}

