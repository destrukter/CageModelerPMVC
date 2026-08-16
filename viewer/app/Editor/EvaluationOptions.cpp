#include <Editor/EvaluationOptions.h>

#include <cstdlib>
#include <string_view>

namespace EvaluationOptions
{
	namespace
	{
		constexpr std::string_view kEvaluationConfigFlag = "--eval-config";
		constexpr auto kEvaluationConfigEnvironmentVariable = "CAGEMODELER_EVAL_CONFIG";

		// Windows separates path lists with ';' because ':' follows a drive letter.
#ifdef _WIN32
		constexpr auto kPathListSeparator = ';';
#else
		constexpr auto kPathListSeparator = ':';
#endif

		std::vector<std::string> GEvaluationConfigPaths;

		void SplitPathList(const std::string& value, std::vector<std::string>& outPaths)
		{
			std::string::size_type begin = 0;
			while (begin <= value.size())
			{
				const auto end = value.find(kPathListSeparator, begin);
				auto entry = value.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
				if (!entry.empty())
				{
					outPaths.push_back(std::move(entry));
				}

				if (end == std::string::npos)
				{
					break;
				}

				begin = end + 1;
			}
		}
	}

	void ParseCommandLine(const int argc, char** argv)
	{
		for (auto i = 1; i < argc; ++i)
		{
			const std::string_view argument(argv[i]);
			if (!argument.starts_with(kEvaluationConfigFlag))
			{
				continue;
			}

			const auto remainder = argument.substr(kEvaluationConfigFlag.size());
			if (remainder.empty())
			{
				// "--eval-config <path>", the path is the next argument.
				if (i + 1 < argc)
				{
					GEvaluationConfigPaths.emplace_back(argv[++i]);
				}
			}
			else if (remainder.front() == '=')
			{
				GEvaluationConfigPaths.emplace_back(remainder.substr(1));
			}
		}
	}

	std::vector<std::string> GetEvaluationConfigPaths()
	{
		if (!GEvaluationConfigPaths.empty())
		{
			return GEvaluationConfigPaths;
		}

		std::vector<std::string> fromEnvironment;
		if (const auto* const value = std::getenv(kEvaluationConfigEnvironmentVariable))
		{
			SplitPathList(value, fromEnvironment);
		}

		return fromEnvironment;
	}
}
