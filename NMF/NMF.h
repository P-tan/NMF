#pragma once
#include <cassert>
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
public:
	bool IsConverged(
		const Mat &X,
		const Mat &U,
		const Mat &V,
		int loop_count
		)
	{
		if (loop_count > 100) {
			return true;
		}
		const double l2norm = (X - U * V).norm();
		const double eps = 1e-7;
		if (l2norm < eps)
		{
			return true;
		}
		return false;
	}
};


class NullProgressReporter
{
public:
	void Report(
		const Mat &X,
		const Mat &U,
		const Mat &V,
		int loop_count
		)
	{}
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
	ProgressReporter progressReporter = ProgressReporter()
	)
{
	assert(X.minCoeff() >= 0);

	initializer.Initialize(X, r, U, V);

	int loop_count = 0;
	progressReporter.Report(X, U, V, loop_count);
	do {
		updater.Update(U, V);
		++loop_count;
		progressReporter.Report(X, U, V, loop_count);
	} while (convergenceTester.IsConverged(X, U, V, loop_count));
}

