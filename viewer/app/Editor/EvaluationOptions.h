#pragma once

#include <string>
#include <vector>

/**
 * Process wide options of the evaluation run, so generated evaluation configs can be run
 * without copying them over the default 'evaluation/projects.json'.
 */
namespace EvaluationOptions
{
	/**
	 * Reads the evaluation options out of the command line. Accepts both
	 * "--eval-config <path>" and "--eval-config=<path>", and may be repeated to run
	 * several configs in one launch.
	 */
	void ParseCommandLine(int argc, char** argv);

	/**
	 * @return The evaluation configs to run, in the order they were given, or an empty
	 * vector when the default config should be used. The command line takes precedence
	 * over the CAGEMODELER_EVAL_CONFIG environment variable, which may hold several paths
	 * separated by the platform's path list separator (';' on Windows, ':' elsewhere).
	 * Entries naming a directory are the caller's to expand.
	 */
	[[nodiscard]] std::vector<std::string> GetEvaluationConfigPaths();
}
