#include <Editor/Editor.h>
#include <Editor/EvaluationOptions.h>
#include <Editor/Scene.h>
#include <Input/InputSubsystem.h>
#include <Mesh/Operations/MeshOperationSystem.h>
#include <Mesh/Operations/MeshComputeDeformationOperation.h>
#include <Mesh/Operations/MeshExportInfluenceMapOperation.h>
#include <Mesh/Operations/MeshExportDistanceFieldOperation.h>
#include <Mesh/Operations/MeshComputeInfluenceMapOperation.h>
#include <Mesh/Operations/MeshExportWeightsOperation.h>
#include <Mesh/Operations/MeshComputeWeightsOperation.h>
#include <Mesh/Operations/MeshExportOperation.h>
#include <Mesh/Operations/MeshLoadOperation.h>
#include <Mesh/MeshLibrary.h>
#include <Navigation/CameraSubsystem.h>
#include <UI/UIStyle.h>
#include <UI/StatusBar.h>
#include <UI/ToolBar.h>
#include <UI/ProjectOptionsPanel.h>
#include <UI/ProjectSettingsPanel.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <thread>
#include <filesystem>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <future>

#include <regex>

namespace
{
	constexpr auto VertexSelectionMinDistanceThresholdSq = 50.0f;
	constexpr auto EdgeSelectionMinDistanceThreshold = 20.0f;
	constexpr auto PolygonSelectionMinDistanceThreshold = 5.0f;

	/**
	 * Draws a selection rectangle on the screen when holding the LMB.
	 * @return A valid tuple of start and end screen space positions when the mouse has been released.
	 */
	void DrawSelectionRect(const ImVec2& startPosition, const ImVec2& endPosition)
	{
		const auto drawList = ImGui::GetForegroundDrawList();
		drawList->AddRect(startPosition, endPosition, ImGui::GetColorU32(IM_COL32(25, 175, 120, 255)));
		drawList->AddRectFilled(startPosition, endPosition, ImGui::GetColorU32(IM_COL32(25, 175, 120, 20)));
	}

	[[nodiscard]] inline TransformationAxis GetTransformationAxisFromGizmoAxis(const GizmoAxis axis)
	{
		switch (axis)
		{
			case GizmoAxis::X:
				return TransformationAxis::X;
			case GizmoAxis::Y:
				return TransformationAxis::Y;
			case GizmoAxis::Z:
				return TransformationAxis::Z;
			default:
				break;
		}

		CheckNoEntry("Invalid axis.");

		return TransformationAxis::X;
	}

	[[nodiscard]] inline TransformationType GetTransformationTypeFromGizmoType(const GizmoType type)
	{
		switch (type)
		{
			case GizmoType::Translate:
				return TransformationType::Translate;
			case GizmoType::Rotate:
				return TransformationType::Rotate;
			case GizmoType::Scale:
				return TransformationType::Scale;
			default:
				break;
		}

		CheckNoEntry("Invalid type.");

		return TransformationType::Translate;
	}

	[[nodiscard]] inline bool HasModifierKeysPressed(const SDL_Keymod modifierKeys)
	{
		return IsSet(modifierKeys, SDL_KMOD_LSHIFT) || IsSet(modifierKeys, SDL_KMOD_LALT) || IsSet(modifierKeys, SDL_KMOD_LGUI);
	}

	[[nodiscard]] std::string ToLower(std::string value)
	{
		std::transform(value.begin(), value.end(), value.begin(), [](const unsigned char c) { return static_cast<char>(std::tolower(c)); });
		return value;
	}

	[[nodiscard]] std::optional<DeformationType> ParseDeformationType(const std::string& value);

	[[nodiscard]] std::string EscapeJsonString(const std::string& value)
	{
		std::string escaped;
		escaped.reserve(value.size());
		for (const char c : value)
		{
			switch (c)
			{
			case '\\': escaped += "\\\\"; break;
			case '"': escaped += '\\'; escaped += '"'; break;
			case '\n': escaped += "\\n"; break;
			case '\r': escaped += "\\r"; break;
			case '\t': escaped += "\\t"; break;
			default: escaped += c; break;
			}
		}
		return escaped;
	}

	struct EvaluationProjectConfig
	{
		std::string _name;
		std::string _coordinateType = "MVC";
		std::string _mesh;
		std::string _cage;
		std::string _deformedCage;
		std::optional<std::string> _embedding;
		std::optional<int32_t> _samples;

		/// The PMVC settings, mirroring the project settings panel: the peeling depth
		/// ("hitCount"), whether the entry hits take part in the per-ray weight split
		/// ("skipEvenHits") and whether the interior distance biases it
		/// ("useInteriorDistance"). All three only apply to the PMVC coordinate type, PMVCO
		/// weights by solid angle alone and runs with a single hit.
		std::optional<int32_t> _pmvcHitCount;
		std::optional<bool> _pmvcSkipEvenHits;
		std::optional<bool> _pmvcUseInteriorDistance;

		/// The three color map exports are independent of each other and carry their own
		/// vertex selection, so a single project can export all of them at once.
		///
		/// The influence map ("influenceMap") is exported for any number of marked cage
		/// vertices ("influenceVertices"), while both distance maps are measured from
		/// exactly one cage vertex each ("euclideanDistanceVertex" /
		/// "interiorDistanceVertex").
		///
		/// The interior distances are read back from the table the interior distance PMVC
		/// variant fills, so "interiorDistanceMap" only produces an export for a PMVC
		/// project that runs with "useInteriorDistance" and at least two hits.
		std::optional<bool> _influenceMap;
		std::optional<std::vector<int32_t>> _influenceVertices;
		std::optional<bool> _euclideanDistanceMap;
		std::optional<int32_t> _euclideanDistanceVertex;
		std::optional<bool> _interiorDistanceMap;
		std::optional<int32_t> _interiorDistanceVertex;

		/// "distanceFieldInterval" is the isoline spacing in world units (unset spaces them
		/// automatically, zero draws none), "distanceFieldEmphasis" emphasizes every n-th
		/// isoline and "distanceFieldMax" fixes the normalization maximum so that the
		/// colors of separate exports stay comparable. They apply to both distance maps.
		std::optional<float> _distanceFieldInterval;
		std::optional<int32_t> _distanceFieldEmphasis;
		std::optional<float> _distanceFieldMax;

		/// @return The cage vertices the influence map is exported for, if it is enabled.
		[[nodiscard]] std::optional<std::vector<int32_t>> GetInfluenceVertices() const
		{
			return _influenceMap.value_or(false) ? _influenceVertices : std::nullopt;
		}

		/**
		 * @return The single cage vertex a distance map is measured from, if it is enabled.
		 * @param useEuclideanDistance Resolve the Euclidean map instead of the interior one.
		 */
		[[nodiscard]] std::optional<std::vector<int32_t>> GetDistanceFieldVertices(const bool useEuclideanDistance) const
		{
			const auto& toggle = useEuclideanDistance ? _euclideanDistanceMap : _interiorDistanceMap;
			const auto& vertex = useEuclideanDistance ? _euclideanDistanceVertex : _interiorDistanceVertex;

			if (!toggle.value_or(false) || !vertex.has_value())
			{
				return std::nullopt;
			}

			return std::vector<int32_t> { vertex.value() };
		}

		/**
		 * @return Whether a run of this project fills the interior detour table, which is
		 * what the interior distance map is read back from. CubemapManager only builds it
		 * for plain PMVC with the interior distance enabled and at least two hits, because
		 * the first hit of a ray never carries a detour.
		 */
		[[nodiscard]] bool ComputesInteriorDistances(const DeformationType deformationType) const
		{
			return deformationType == DeformationType::PMVC &&
				_pmvcUseInteriorDistance.value_or(false) &&
				_pmvcHitCount.value_or(1) >= 2;
		}
	};

	struct EvaluationConfig
	{
		std::string _timingsFile = "timings.json";
		std::vector<EvaluationProjectConfig> _projects;
	};

	[[nodiscard]] std::optional<std::string> ExtractJsonStringValue(const std::string& objectText, const std::string& key)
	{
		const std::regex pattern("\"" + key + "\"\\s*:\\s*\"([^\"]*)\"");
		std::smatch match;
		if (std::regex_search(objectText, match, pattern) && match.size() > 1)
		{
			return match[1].str();
		}
		return std::nullopt;
	}

	[[nodiscard]] std::optional<bool> ExtractJsonBoolValue(const std::string& objectText, const std::string& key)
	{
		const std::regex pattern("\"" + key + "\"\\s*:\\s*(true|false)");
		std::smatch match;
		if (std::regex_search(objectText, match, pattern) && match.size() > 1)
		{
			return match[1].str() == "true";
		}
		return std::nullopt;
	}

	[[nodiscard]] std::optional<int32_t> ExtractJsonIntValue(const std::string& objectText, const std::string& key)
	{
		const std::regex pattern("\"" + key + "\"\\s*:\\s*(-?[0-9]+)");
		std::smatch match;
		if (std::regex_search(objectText, match, pattern) && match.size() > 1)
		{
			return static_cast<int32_t>(std::stoi(match[1].str()));
		}
		return std::nullopt;
	}

	[[nodiscard]] std::optional<float> ExtractJsonFloatValue(const std::string& objectText, const std::string& key)
	{
		const std::regex pattern("\"" + key + "\"\\s*:\\s*(-?[0-9]+(?:\\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)");
		std::smatch match;
		if (std::regex_search(objectText, match, pattern) && match.size() > 1)
		{
			return std::stof(match[1].str());
		}
		return std::nullopt;
	}

	[[nodiscard]] std::optional<std::vector<int32_t>> ExtractJsonIntArrayValue(const std::string& objectText, const std::string& key)
	{
		const std::regex pattern("\"" + key + "\"\\s*:\\s*\\[([^\\]]*)\\]");
		std::smatch match;
		if (!std::regex_search(objectText, match, pattern) || match.size() <= 1)
		{
			return std::nullopt;
		}

		std::vector<int32_t> values;
		const auto content = match[1].str();
		const std::regex intPattern("-?[0-9]+");
		auto begin = std::sregex_iterator(content.begin(), content.end(), intPattern);
		auto end = std::sregex_iterator();
		for (auto it = begin; it != end; ++it)
		{
			values.push_back(static_cast<int32_t>(std::stoi(it->str())));
		}

		return values;
	}

	/// Writes the vertex indices a color map was exported for next to the export itself, so
	/// it can be traced back to its selection.
	void WriteSelectedVerticesFile(const std::filesystem::path& filepath, const std::vector<int32_t>& selectedVertices)
	{
		std::ofstream verticesOutput(filepath, std::ios::out | std::ios::trunc);
		if (!verticesOutput.is_open())
		{
			LOG_WARN("Unable to write the selected vertices file '{}'.", filepath.string());

			return;
		}

		for (const auto vertexIdx : selectedVertices)
		{
			verticesOutput << vertexIdx << "\n";
		}
	}

	[[nodiscard]] std::vector<std::string> ExtractTopLevelObjects(const std::string& arrayText)
	{
		std::vector<std::string> objects;
		int32_t braceDepth = 0;
		bool isInString = false;
		std::size_t objectStart = std::string::npos;
		for (std::size_t i = 0; i < arrayText.size(); ++i)
		{
			const auto c = arrayText[i];
			const auto escaped = (i > 0 && arrayText[i - 1] == '\\');
			if (c == '"' && !escaped)
			{
				isInString = !isInString;
				continue;
			}
			if (isInString)
			{
				continue;
			}
			if (c == '{')
			{
				if (braceDepth == 0)
				{
					objectStart = i;
				}
				++braceDepth;
			}
			else if (c == '}')
			{
				--braceDepth;
				if (braceDepth == 0 && objectStart != std::string::npos)
				{
					objects.push_back(arrayText.substr(objectStart, i - objectStart + 1));
					objectStart = std::string::npos;
				}
			}
		}
		return objects;
	}

	/// @return The text between the brackets of the array stored under a key, if there is one.
	[[nodiscard]] std::optional<std::string> ExtractTopLevelArray(const std::string& text, const std::string& key)
	{
		const auto keyPos = text.find("\"" + key + "\"");
		if (keyPos == std::string::npos)
		{
			return std::nullopt;
		}
		const auto arrayStart = text.find('[', keyPos);
		if (arrayStart == std::string::npos)
		{
			return std::nullopt;
		}
		int32_t bracketDepth = 0;
		bool isInString = false;
		for (std::size_t i = arrayStart; i < text.size(); ++i)
		{
			const auto c = text[i];
			const auto escaped = (i > 0 && text[i - 1] == '\\');
			if (c == '"' && !escaped)
			{
				isInString = !isInString;
				continue;
			}
			if (isInString)
			{
				continue;
			}
			if (c == '[')
			{
				++bracketDepth;
			}
			else if (c == ']')
			{
				--bracketDepth;
				if (bracketDepth == 0)
				{
					return text.substr(arrayStart + 1, i - arrayStart - 1);
				}
			}
		}
		return std::nullopt;
	}

	[[nodiscard]] std::optional<std::string> ReadTextFile(const std::filesystem::path& filepath)
	{
		std::ifstream file(filepath);
		if (!file.is_open())
		{
			return std::nullopt;
		}

		return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
	}

	/**
	 * Resolves the evaluation configs to run. A config holds a "projects" array and is run
	 * as is. The manifest gen_eval_configs.py writes next to its configs holds a "configs"
	 * array instead, and every config it lists is run in order, resolved relative to the
	 * manifest, so one launch covers everything the generator produced.
	 */
	[[nodiscard]] std::vector<std::filesystem::path> ResolveEvaluationConfigs(const std::filesystem::path& configPath)
	{
		const auto text = ReadTextFile(configPath);
		if (!text.has_value())
		{
			LOG_ERROR("Unable to open evaluation config '{}'.", configPath.string());
			return { };
		}

		if (text->find("\"projects\"") != std::string::npos)
		{
			return { configPath };
		}

		const auto configsText = ExtractTopLevelArray(text.value(), "configs");
		if (!configsText.has_value())
		{
			LOG_ERROR("Evaluation config '{}' has neither a 'projects' nor a 'configs' array.", configPath.string());
			return { };
		}

		std::vector<std::filesystem::path> configPaths;
		for (const auto& entryText : ExtractTopLevelObjects(configsText.value()))
		{
			if (const auto config = ExtractJsonStringValue(entryText, "config"); config.has_value())
			{
				configPaths.push_back(configPath.parent_path() / config.value());
			}
		}

		LOG_INFO("Evaluation manifest '{}' lists {} configs.", configPath.string(), configPaths.size());

		return configPaths;
	}

	[[nodiscard]] std::optional<EvaluationConfig> ParseEvaluationConfig(const std::string& text)
	{
		EvaluationConfig config;
		if (const auto timingsFile = ExtractJsonStringValue(text, "timingsFile"))
		{
			config._timingsFile = timingsFile.value();
		}
		const auto projectsText = ExtractTopLevelArray(text, "projects");
		if (!projectsText.has_value())
		{
			return std::nullopt;
		}
		for (const auto& objectText : ExtractTopLevelObjects(projectsText.value()))
		{
			EvaluationProjectConfig project;
			project._name = ExtractJsonStringValue(objectText, "name").value_or("");
			project._coordinateType = ExtractJsonStringValue(objectText, "coordinateType").value_or("MVC");
			project._mesh = ExtractJsonStringValue(objectText, "mesh").value_or("");
			project._cage = ExtractJsonStringValue(objectText, "cage").value_or("");
			project._deformedCage = ExtractJsonStringValue(objectText, "deformedCage").value_or("");
			project._embedding = ExtractJsonStringValue(objectText, "embedding");
			project._samples = ExtractJsonIntValue(objectText, "samples");
			project._pmvcHitCount = ExtractJsonIntValue(objectText, "hitCount");
			project._pmvcSkipEvenHits = ExtractJsonBoolValue(objectText, "skipEvenHits");
			project._pmvcUseInteriorDistance = ExtractJsonBoolValue(objectText, "useInteriorDistance");

			project._influenceMap = ExtractJsonBoolValue(objectText, "influenceMap");
			project._influenceVertices = ExtractJsonIntArrayValue(objectText, "influenceVertices");
			project._euclideanDistanceMap = ExtractJsonBoolValue(objectText, "euclideanDistanceMap");
			project._euclideanDistanceVertex = ExtractJsonIntValue(objectText, "euclideanDistanceVertex");
			project._interiorDistanceMap = ExtractJsonBoolValue(objectText, "interiorDistanceMap");
			project._interiorDistanceVertex = ExtractJsonIntValue(objectText, "interiorDistanceVertex");

			project._distanceFieldInterval = ExtractJsonFloatValue(objectText, "distanceFieldInterval");
			project._distanceFieldEmphasis = ExtractJsonIntValue(objectText, "distanceFieldEmphasis");
			project._distanceFieldMax = ExtractJsonFloatValue(objectText, "distanceFieldMax");

			// Keys of the schema that predates the independent map toggles are no longer
			// read, so a config still using them would silently lose its exports.
			for (const auto* const supersededKey : { "vertices", "selectedVertices", "distanceField", "distanceFieldEuclidean" })
			{
				if (objectText.find(std::string("\"") + supersededKey + "\"") != std::string::npos)
				{
					LOG_WARN("Project '{}' uses the superseded key '{}', which is ignored. Use 'influenceMap'/'influenceVertices', "
						"'euclideanDistanceMap'/'euclideanDistanceVertex' and 'interiorDistanceMap'/'interiorDistanceVertex' instead.",
						project._name,
						supersededKey);
				}
			}

			config._projects.push_back(std::move(project));
		}
		return config;
	}

	[[nodiscard]] bool ValidateEvaluationProjectConfig(const EvaluationProjectConfig& project, const std::size_t projectIndex)
	{
		if (project._mesh.empty() || project._cage.empty() || project._deformedCage.empty())
		{
			LOG_WARN("Skipping project {} due to missing required file keys (mesh/cage/deformedCage).", projectIndex);
			return false;
		}

		const auto deformationType = ParseDeformationType(project._coordinateType);
		if (!deformationType.has_value())
		{
			LOG_WARN("Skipping project {} due to unsupported coordinateType '{}'.", projectIndex, project._coordinateType);
			return false;
		}

		// Every hit occupies six cubemap array layers, so the peeling depth is bounded by
		// the array layers Vulkan guarantees rather than clamped behind the config's back.
		if (project._pmvcHitCount.has_value() &&
			(project._pmvcHitCount.value() <= 0 || static_cast<uint64_t>(project._pmvcHitCount.value()) > PMVCSettings::kMaxHitCount))
		{
			LOG_WARN("Skipping project {} due to invalid hitCount {} (must be within 1..{}).",
				projectIndex,
				project._pmvcHitCount.value(),
				PMVCSettings::kMaxHitCount);
			return false;
		}

		// The PMVC settings shape the per-ray weight split, which only the plain PMVC type
		// has: every other type ignores them, and PMVCO has them overridden by its preset.
		const auto hasSplitSettings = project._pmvcHitCount.value_or(1) != 1 ||
			project._pmvcSkipEvenHits.value_or(false) ||
			project._pmvcUseInteriorDistance.value_or(false);
		const auto hasPMVCSettings = hasSplitSettings || project._pmvcHitCount.has_value();
		if (*deformationType == DeformationType::PMVCO && hasSplitSettings)
		{
			LOG_WARN("Project {} sets 'hitCount', 'skipEvenHits' or 'useInteriorDistance' for PMVCO, which weights by solid angle alone "
				"and always runs with a single hit. The settings are ignored.", projectIndex);
		}
		else if (!DeformationTypeHelpers::IsPMVC(*deformationType) && hasPMVCSettings)
		{
			LOG_WARN("Project {} sets PMVC settings for coordinateType '{}', which ignores them.", projectIndex, project._coordinateType);
		}

		// The first hit of a ray never carries a detour, so with a single hit the interior
		// distance cannot change any weight and no detour table is built.
		if (*deformationType == DeformationType::PMVC &&
			project._pmvcUseInteriorDistance.value_or(false) &&
			project._pmvcHitCount.value_or(1) < 2)
		{
			LOG_WARN("Project {} enables 'useInteriorDistance' with a single hit, where it has no effect. Raise 'hitCount' to at least 2.", projectIndex);
		}

		if (project._influenceMap.value_or(false) && !project.GetInfluenceVertices().has_value())
		{
			LOG_WARN("Skipping project {} because 'influenceMap' is enabled but no 'influenceVertices' were given.", projectIndex);
			return false;
		}

		// Both distance maps are measured from exactly one cage vertex, so an enabled map
		// without a vertex cannot be exported at all.
		if (project._euclideanDistanceMap.value_or(false) && !project.GetDistanceFieldVertices(true).has_value())
		{
			LOG_WARN("Skipping project {} because 'euclideanDistanceMap' is enabled but no 'euclideanDistanceVertex' was given.", projectIndex);
			return false;
		}

		if (project._interiorDistanceMap.value_or(false) && !project.GetDistanceFieldVertices(false).has_value())
		{
			LOG_WARN("Skipping project {} because 'interiorDistanceMap' is enabled but no 'interiorDistanceVertex' was given.", projectIndex);
			return false;
		}

		if (project._interiorDistanceMap.value_or(false) && !project.ComputesInteriorDistances(*deformationType))
		{
			LOG_WARN("Project {} enables 'interiorDistanceMap', but only PMVC with 'useInteriorDistance' and a 'hitCount' of at least 2 "
				"computes interior distances. The export will be skipped.", projectIndex);
		}

		return true;
	}

	[[nodiscard]] std::optional<DeformationType> ParseDeformationType(const std::string& value)
	{
		static const std::unordered_map<std::string, DeformationType> mapping = {
			{ "mvc", DeformationType::MVC },
			{ "qmvc", DeformationType::QMVC },
			{ "harmonic", DeformationType::Harmonic },
			{ "bbw", DeformationType::BBW },
			{ "lbc", DeformationType::LBC },
			{ "mec", DeformationType::MEC },
			{ "mlc", DeformationType::MLC },
			{ "green", DeformationType::Green },
			{ "qgc", DeformationType::QGC },
			{ "somigliana", DeformationType::Somigliana },
			{ "pmvc", DeformationType::PMVC },
			{ "pmvco", DeformationType::PMVCO }
		};

		const auto it = mapping.find(ToLower(value));
		if (it == mapping.end())
		{
			return std::nullopt;
		}

		return it->second;
	}
}

Editor::Editor(const SubsystemPtr<InputSubsystem>& inputSubsystem,
	const SubsystemPtr<CameraSubsystem>& cameraSubsystem)
	: _inputSubsystem(inputSubsystem)
	, _cameraSubsystem(cameraSubsystem)
	, _isGizmoHighlighted(false)
	, _isGizmoTransformed(false)
	, _isDragging(false)
	, _isSelectingRect(false)
	, _hasDragged(false)
{
	inputSubsystem->RegisterInputActionMapping(InputActionMapping { "EditorClick", SDL_KMOD_NONE, SDL_BUTTON_LEFT, { }});
	inputSubsystem->RegisterInputActionEntry(InputActionEntry { "EditorClick",
		[this]<typename ParamsType>(ParamsType&& actionParams)
		{
			OnClicked(std::forward<ParamsType>(actionParams));
		}});
}

void Editor::Initialize(const std::shared_ptr<SceneRenderer>& sceneRenderer, const std::shared_ptr<CubemapManager>& cubemapRenderer)
{
	_scene = std::make_unique<Scene>(sceneRenderer);
	_cubemapRenderer = cubemapRenderer;

	// Sets up all the scene lights before initializing the renderer. Hacky!
	CreateSceneLights();

	sceneRenderer->Initialize();

	_meshOperationSystem = std::make_shared<MeshOperationSystem>();
	_projectData = std::make_shared<ProjectData>();
	_threadPool = std::make_unique<ThreadPool>(2);
	_mainThreadQueue = std::make_unique<ThreadSafeQueue<FunctionWrapper>>();

	_toolSystem = std::make_shared<ToolSystem>();
	_toolSystem->SetSelectionChangedDelegate([this]<typename T>(T&& toolType) {
		OnToolSelectionChanged(std::forward<T>(toolType));
	});

	// Adjusts the UI style and set up all UI elements.
	UIStyle::SetStyle();
	SetUpUIElements();

	// Create the scene gizmo.
	const auto& viewInfo = _cameraSubsystem->GetCamera().GetViewInfo();
	_gizmo = std::make_shared<Gizmo>(*_scene);
	_gizmo->UpdateModelMatrix(viewInfo);

//#if BUILD_DEVELOPMENT
	_newProjectPanel = std::make_shared<NewProjectPanel>(_meshOperationSystem,
		[this] { OnNewProjectCancelled(); },
		[this] { OnNewProjectCreated(); });

	_projectModel->_deformationType = DeformationType::PMVC;
	// Peel three layers in the default startup project so the per-ray weight split is
	// exercised on launch instead of degenerating to the single-hit case.
	_projectModel->_pmvcHitCount = 3;
	_projectModel->_pmvcSkipEvenHits = false;
	_projectModel->_pmvcUseInteriorDistance = false;
	_projectModel->ApplyPMVCPreset();
	//_projectModel->_meshFilepath = "assets/meshes/tri.obj";
	//_projectModel->_cageFilepath = "assets/meshes/sphere_cages_triangulated.obj";
	_projectModel->_meshFilepath = "assets/meshes/armadilloman.obj";
	_projectModel->_cageFilepath = "assets/meshes/armadilloman_cages_triangulated.obj";
	_projectModel->_embeddingFilepath = "assets/meshes/bishop_cages_triangulated_embedding.msh";
	_projectModel->_deformedCageFilepath = "assets/meshes/armadilloman_cages_triangulated_deformed_8.obj";
	_newProjectPanel->SetModel(_projectModel);
	_projectOptionsPanel->SetModelData(_projectModel);

	StartEvaluation();
	OnNewProjectCreated();
//#endif
}

void Editor::StartEvaluation()
{
	const auto kEvaluationRoot = std::filesystem::path("evaluation");
	const auto evaluationRoot = std::filesystem::absolute(kEvaluationRoot);

	// What runs, in order of precedence: the config or manifest given with
	// "--eval-config <path>" or CAGEMODELER_EVAL_CONFIG, then the manifest of the
	// generated configs, then the hand written projects.json. Paths inside a config stay
	// relative to the evaluation directory regardless of where the config itself is.
	std::filesystem::path configPath;
	if (const auto configOverride = EvaluationOptions::GetEvaluationConfigPath(); !configOverride.empty())
	{
		configPath = std::filesystem::absolute(configOverride);
	}
	else if (std::filesystem::exists(evaluationRoot / "generated" / "manifest.json"))
	{
		configPath = evaluationRoot / "generated" / "manifest.json";
	}
	else
	{
		configPath = evaluationRoot / "projects.json";
	}

	if (!std::filesystem::exists(configPath))
	{
		LOG_WARN("Evaluation config '{}' does not exist (working directory '{}'). Skipping evaluation run.",
			configPath.string(),
			std::filesystem::current_path().string());
		return;
	}

	LOG_INFO("Running evaluation config '{}'.", configPath.string());

	for (const auto& resolvedConfigPath : ResolveEvaluationConfigs(configPath))
	{
		RunEvaluationConfig(evaluationRoot, resolvedConfigPath);
	}
}

void Editor::RunEvaluationConfig(const std::filesystem::path& evaluationRoot, const std::filesystem::path& configPath)
{
	if (!std::filesystem::exists(configPath))
	{
		LOG_WARN("Evaluation config '{}' does not exist. Skipping it.", configPath.string());
		return;
	}

	const auto configContent = ReadTextFile(configPath);
	if (!configContent.has_value())
	{
		LOG_ERROR("Unable to open evaluation config '{}'.", configPath.string());
		return;
	}

	const auto parsedConfig = ParseEvaluationConfig(configContent.value());
	if (!parsedConfig.has_value())
	{
		LOG_ERROR("Failed to parse evaluation config '{}'.", configPath.string());
		return;
	}

	const auto timingOutputPath = evaluationRoot / parsedConfig->_timingsFile;

	// A generated config points its timings at a per-config file inside a results
	// directory, which only exists once something has created it.
	if (timingOutputPath.has_parent_path())
	{
		std::filesystem::create_directories(timingOutputPath.parent_path());
	}

#ifdef NDEBUG
	constexpr auto buildType = "Release";
#else
	constexpr auto buildType = "Debug/Development";
#endif

	/// The PMVC settings a project actually ran with, after ApplyPMVCPreset() and the
	/// interior distance fallback, which can differ from what the config asked for.
	struct EvaluationPMVCSettings
	{
		uint64_t _hitCount = 1;
		bool _skipEvenHits = false;
		bool _useInteriorDistance = false;
	};

	struct EvaluationResult
	{
		std::string _projectName;
		std::string _coordinateType;
		std::string _status = "FAILED";
		std::optional<double> _elapsedMs;
		EvaluationStageTimings _stageTimings;
		std::optional<int32_t> _meshVertexCount;
		std::optional<int32_t> _cageVertexCount;

		/// Only recorded for the plain PMVC type, the only one with a per-ray weight split.
		std::optional<EvaluationPMVCSettings> _pmvcSettings;
	};

	std::vector<EvaluationResult> results;
	results.reserve(parsedConfig->_projects.size());

	
	_isEvaluationMode = true;

	for (std::size_t i = 0; i < parsedConfig->_projects.size(); ++i)
	{
		const auto& project = parsedConfig->_projects[i];
		// A single invalid project skips that project, it does not abandon the rest of the
		// batch. ValidateEvaluationProjectConfig has already logged why.
		if (!ValidateEvaluationProjectConfig(project, i))
		{
			continue;
		}

		// Validated above, so the coordinate type is known.
		const auto deformationType = ParseDeformationType(project._coordinateType);

		const auto projectName = project._name.empty() ? (std::string("project_") + std::to_string(i)) : project._name;
		const auto projectOutputDir = evaluationRoot / projectName;
		std::filesystem::create_directories(projectOutputDir);

		_projectModel->_deformationType = *deformationType;
		_projectModel->_meshFilepath = evaluationRoot / project._mesh;
		_projectModel->_cageFilepath = evaluationRoot / project._cage;
		_projectModel->_deformedCageFilepath = evaluationRoot / project._deformedCage;

		// Reset rather than inherit: leaving the embedding of the previous project (or of
		// the startup project) in place would silently evaluate the wrong one.
		_projectModel->_embeddingFilepath = project._embedding.has_value()
			? std::optional<std::filesystem::path>(evaluationRoot / project._embedding.value())
			: std::nullopt;
		if (DeformationTypeHelpers::RequiresEmbedding(*deformationType) && !_projectModel->_embeddingFilepath.has_value())
		{
			LOG_WARN("Project '{}' uses coordinateType '{}', which requires an embedding, but the config has no 'embedding' key.",
				projectName,
				project._coordinateType);
		}

		if (project._samples.has_value())
		{
			_projectModel->_numSamples = project._samples.value();
		}

		_projectModel->_pmvcHitCount = static_cast<uint64_t>(project._pmvcHitCount.value_or(1));
		_projectModel->_pmvcSkipEvenHits = project._pmvcSkipEvenHits.value_or(false);
		_projectModel->_pmvcUseInteriorDistance = project._pmvcUseInteriorDistance.value_or(false);
		// The offset variant is the PMVCO coordinate type, so the preset derives it (and
		// the settings it implies) from the coordinate type of the project.
		_projectModel->ApplyPMVCPreset();
		// With a single hit the interior distance is skipped by the renderer, so the
		// record reports what actually ran rather than what was requested.
		if (_projectModel->_pmvcHitCount < 2)
		{
			_projectModel->_pmvcUseInteriorDistance = false;
		}

		EvaluationResult result;
		result._projectName = projectName;
		result._coordinateType = project._coordinateType;
		if (*deformationType == DeformationType::PMVC)
		{
			result._pmvcSettings = EvaluationPMVCSettings {
				_projectModel->_pmvcHitCount,
				_projectModel->_pmvcSkipEvenHits,
				_projectModel->_pmvcUseInteriorDistance
			};
		}

		const auto start = std::chrono::steady_clock::now();

		_projectCreationFailed.store(false, std::memory_order_seq_cst);
		{
			std::scoped_lock lock(_evaluationTimingsMutex);
			_latestEvaluationStageTimings = {};
		}

		auto completionPromise = std::make_shared<std::promise<void>>();
		auto completionFuture = completionPromise->get_future();

		OnNewProjectCreated(completionPromise);

		while (completionFuture.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready)
		{
			if (_projectCreationFailed.load(std::memory_order_seq_cst))
			{
				break;
			}

			FunctionWrapper mainThreadFunction;
			while (_mainThreadQueue->TryPop(mainThreadFunction))
			{
				mainThreadFunction();
			}

			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}

		FunctionWrapper mainThreadFunction;
		while (_mainThreadQueue->TryPop(mainThreadFunction))
		{
			mainThreadFunction();
		}

		if (_projectCreationFailed.load(std::memory_order_seq_cst))
		{
			LOG_WARN("Evaluation project '{}' failed.", projectName);

			results.push_back(std::move(result));
			continue;
		}

		const auto end = std::chrono::steady_clock::now();
		const auto elapsedMs = std::chrono::duration<double, std::milli>(end - start).count();

		EvaluationStageTimings stageTimings;
		{
			std::scoped_lock lock(_evaluationTimingsMutex);
			stageTimings = _latestEvaluationStageTimings;
		}

		ExportWeights(projectOutputDir / "weights.dmat");
		ExportDeformedCage(projectOutputDir / "deformed_cage.obj");
		ExportDeformedMeshes(projectOutputDir / "deformed_mesh.obj");
		if (const auto influenceVertices = project.GetInfluenceVertices(); influenceVertices.has_value())
		{
			LOG_DEBUG("Exporting influence map for project '{}' with {} vertices.", projectName, influenceVertices->size());
			ExportInfluenceColorMap(projectOutputDir / "influence_map.obj", influenceVertices);
			WriteSelectedVerticesFile(projectOutputDir / "influence_map_vertices.txt", influenceVertices.value());
		}

		// The two distance maps follow the influence map (same vertex colored format) but
		// are toggled independently of it and of each other, each measured from its own
		// single cage vertex, so one project can export both.
		for (const auto useEuclideanDistance : { true, false })
		{
			const auto distanceVertices = project.GetDistanceFieldVertices(useEuclideanDistance);
			if (!distanceVertices.has_value())
			{
				continue;
			}

			// The detour table is memoized across projects, so exporting it for a run that
			// did not fill it would silently export the table of an earlier project.
			if (!useEuclideanDistance && !project.ComputesInteriorDistances(*deformationType))
			{
				continue;
			}

			// Keeps the defaults of the color map (automatic isoline spacing, normalized
			// against the maximum of the data set) for every key the project omits.
			DistanceColorMapParams colorMapParams;
			if (project._distanceFieldInterval.has_value())
			{
				colorMapParams.contourInterval = project._distanceFieldInterval.value();
			}
			if (project._distanceFieldEmphasis.has_value())
			{
				colorMapParams.contourEmphasisEvery = project._distanceFieldEmphasis.value();
			}
			if (project._distanceFieldMax.has_value())
			{
				colorMapParams.maxDistance = project._distanceFieldMax.value();
			}

			const std::string fileStem = useEuclideanDistance ? "euclidean_distance_map" : "interior_distance_map";
			LOG_DEBUG("Exporting the {} distance field for project '{}'.", useEuclideanDistance ? "euclidean" : "interior", projectName);
			ExportDistanceFieldColorMap(projectOutputDir / (fileStem + ".obj"),
				useEuclideanDistance,
				distanceVertices,
				std::move(colorMapParams));
			WriteSelectedVerticesFile(projectOutputDir / (fileStem + "_vertices.txt"), distanceVertices.value());
		}
		const auto meshVertexCount = _projectData
			? std::optional<int32_t>(static_cast<int32_t>(_projectData->_mesh._vertices.rows()))
			: std::nullopt;
		const auto cageVertexCount = _projectData
			? std::optional<int32_t>(static_cast<int32_t>(_projectData->_cage._vertices.rows()))
			: std::nullopt;
		//_cubemapRenderer->Cleanup();
		ClearEvaluationData();
		
		//_cubemapRenderer->~CubemapManager();
		//_cubemapRenderer = std::make_shared<CubemapManager>(_sceneManager->_renderPipelineManager, _renderResourceManager, _device, _instance, 32, VK_FORMAT_R32G32B32A32_SFLOAT);
		

		result._status = "OK";
		result._elapsedMs = elapsedMs;
		result._stageTimings = stageTimings;
		result._meshVertexCount = meshVertexCount;
		result._cageVertexCount = cageVertexCount;
		results.push_back(std::move(result));
		LOG_INFO("Evaluation project '{}' finished in {} ms.", projectName, elapsedMs);
	}

	_isEvaluationMode = false;
	std::ofstream timingOutput(timingOutputPath, std::ios::out | std::ios::trunc);
	if (!timingOutput.is_open())
	{
		LOG_ERROR("Unable to open timing output file '{}'.", timingOutputPath.string());
		return;
	}

	timingOutput << "{\n";
	timingOutput << "  \"buildType\": \"" << EscapeJsonString(buildType) << "\",\n";
	timingOutput << "  \"projectCount\": " << parsedConfig->_projects.size() << ",\n";
	timingOutput << "  \"results\": [\n";
	for (std::size_t i = 0; i < results.size(); ++i)
	{
		const auto& result = results[i];
		timingOutput << "    {\n";
		timingOutput << "      \"projectName\": \"" << EscapeJsonString(result._projectName) << "\",\n";
		timingOutput << "      \"coordinateType\": \"" << EscapeJsonString(result._coordinateType) << "\",\n";
		timingOutput << "      \"status\": \"" << EscapeJsonString(result._status) << "\"";
		if (result._elapsedMs.has_value())
		{
			timingOutput << ",\n      \"elapsedMs\": " << result._elapsedMs.value();
		}
		const auto writeTiming = [&timingOutput](const char* key, const std::optional<double>& value)
		{
			if (value.has_value())
			{
				timingOutput << ",\n      \"" << key << "\": " << value.value();
			}
		};
		writeTiming("initMs", result._stageTimings._initMs);
		writeTiming("renderMs", result._stageTimings._renderMs);
		writeTiming("transferMs", result._stageTimings._transferMs);
		writeTiming("computeMs", result._stageTimings._computeMs);
		writeTiming("computeTotalMs", result._stageTimings._computeTotalMs);
		writeTiming("deformationApplyMs", result._stageTimings._deformationApplyMs);
		if (result._meshVertexCount.has_value())
		{
			timingOutput << ",\n      \"numMeshVertices\": " << result._meshVertexCount.value();
		}
		if (result._cageVertexCount.has_value())
		{
			timingOutput << ",\n      \"numCageVertices\": " << result._cageVertexCount.value();
		}
		if (result._pmvcSettings.has_value())
		{
			const auto& settings = result._pmvcSettings.value();
			timingOutput << ",\n      \"hitCount\": " << settings._hitCount;
			timingOutput << ",\n      \"skipEvenHits\": " << (settings._skipEvenHits ? "true" : "false");
			timingOutput << ",\n      \"useInteriorDistance\": " << (settings._useInteriorDistance ? "true" : "false");
		}
		timingOutput << "\n    }" << (i + 1 < results.size() ? "," : "") << "\n";
	}
	timingOutput << "  ]\n";
	timingOutput << "}\n";
}

void Editor::CreateSceneLights() const
{
	_scene->AddLightSource(PointLight(glm::vec3(4.0f, 4.0f, 4.0f), 0.75f));
	_scene->AddLightSource(PointLight(glm::vec3(-3.0f, -7.5f, 6.0f), 1.08f));
	_scene->AddLightSource(PointLight(glm::vec3(0.0f, 8.0f, 6.0f), 0.92f));
	_scene->AddLightSource(PointLight(glm::vec3(8.0f, -8.0f, 8.0f), 1.52f));
}

void Editor::RecordUI()
{
	if (ImGui::BeginMainMenuBar())
	{
		if (ImGui::BeginMenu("File"))
		{
			if (ImGui::MenuItem("New Project..."))
			{
				_newProjectPanel = std::make_shared<NewProjectPanel>(_meshOperationSystem,
					[this] { OnNewProjectCancelled(); },
					[this] { OnNewProjectCreated(); });
				_newProjectPanel->Present();
			}

			const auto weightsData = _weightsData.LockRead();
			const auto hasComputedWeights = (weightsData->_weights.cols() > 0 &&
				!_isComputingWeightsData.load(std::memory_order_relaxed) &&
				!_isComputingDeformationData.load(std::memory_order_relaxed));

			ImGui::BeginDisabled(!hasComputedWeights);
			{
				if (ImGui::BeginMenu("Export"))
				{
					if (ImGui::MenuItem("Current Frame Mesh...", nullptr))
					{
						const auto filepath = UIHelpers::PresentExportFilePopup({ { "Mesh (.obj)", "obj" } }, "Untitled.obj");

						if (filepath.has_value())
						{
							ExportCurrentDeformedMesh(filepath.value());
						}
					}

					if (ImGui::MenuItem("Meshes...", nullptr))
					{
						const auto filepath = UIHelpers::PresentExportFilePopup({ { "Mesh (.obj)", "obj" } }, "Untitled.obj");

						if (filepath.has_value())
						{
							ExportDeformedMeshes(filepath.value());
						}
					}

					if (ImGui::MenuItem("Deformed Cage...", nullptr))
					{
						const auto filepath = UIHelpers::PresentExportFilePopup({ { "Mesh (.obj)", "obj" } }, "Untitled.obj");

						if (filepath.has_value())
						{
							ExportDeformedCage(filepath.value());
						}
					}

					if (ImGui::MenuItem("Influence Color Map...", nullptr))
					{
						const auto filepath = UIHelpers::PresentExportFilePopup({ { "Mesh (.obj)", "obj" } }, "Untitled.obj");

						if (filepath.has_value())
						{
							ExportInfluenceColorMap(filepath.value());
						}
					}

					if (ImGui::MenuItem("Interior Distance Color Map...", nullptr))
					{
						const auto filepath = UIHelpers::PresentExportFilePopup({ { "Mesh (.obj)", "obj" } }, "interior_distance_map.obj");

						if (filepath.has_value())
						{
							ExportDistanceFieldColorMap(filepath.value(), false);
						}
					}

					if (ImGui::MenuItem("Euclidean Distance Color Map...", nullptr))
					{
						const auto filepath = UIHelpers::PresentExportFilePopup({ { "Mesh (.obj)", "obj" } }, "euclidean_distance_map.obj");

						if (filepath.has_value())
						{
							ExportDistanceFieldColorMap(filepath.value(), true);
						}
					}

					ImGui::BeginDisabled(_projectModel->_deformationType != DeformationType::BBW && _projectModel->_deformationType != DeformationType::LBC);
					{
						if (ImGui::MenuItem("Weights...", nullptr))
						{
							const auto filepath = UIHelpers::PresentExportFilePopup({ { "Weights (.dmat)", "dmat" } }, "Untitled.dmat");

							if (filepath.has_value())
							{
								ExportWeights(filepath.value());
							}
						}
					}
					ImGui::EndDisabled();

					ImGui::EndMenu();
				}
			}
			ImGui::EndDisabled();

			ImGui::EndMenu();
		}

		if (ImGui::BeginMenu("Edit"))
		{
			if (ImGui::MenuItem("Project Settings..."))
			{
				_projectSettingsPanel = std::make_shared<ProjectSettingsPanel>(_projectModel,
					_meshOperationSystem,
					[this] { OnProjectSettingsCancelled(); },
					[this] { OnProjectSettingsApplied(); });
				_projectSettingsPanel->Present();
			}

			ImGui::EndMenu();
		}

		ImGui::EndMainMenuBar();
	}

	_statusBar->Layout();
	_toolBar->Layout();
	_projectOptionsPanel->Layout();

	if (_newProjectPanel != nullptr)
	{
		_newProjectPanel->Layout();
	}

	if (_projectSettingsPanel != nullptr)
	{
		_projectSettingsPanel->Layout();
	}

	// If we are making a rectangle selection on the screen we want to do it before we process any mesh selection, so we can use the data.
	if (_isSelectingRect)
	{
		DrawSelectionRect(_selectionRectStartPosition, _selectionRectEndPosition);
	}
}

void Editor::Update(const double deltaTime)
{
	FunctionWrapper mainThreadFunction;
	while (_mainThreadQueue->TryPop(mainThreadFunction))
	{
		mainThreadFunction();
	}

	if (_newProjectPanel != nullptr && _newProjectPanel->IsModalPanelVisible())
	{
		return;
	}

	if (_projectSettingsPanel != nullptr && _projectSettingsPanel->IsModalPanelVisible())
	{
		return;
	}

	// Update the matrices of all gizmos.
	const auto& camera = _cameraSubsystem->GetCamera();
	const auto& viewInfo = camera.GetViewInfo();
	_gizmo->UpdateModelMatrix(viewInfo);

	// Shoot a ray to the currently active gizmo type to determine the selection.
	const auto activeGizmoType = _toolBar->GetActiveGizmoType();

	if (activeGizmoType != GizmoType::MaxNum)
	{
		UpdateGizmoSelection(viewInfo, activeGizmoType);
	}

	if (_deformedMeshHandle == InvalidHandle || _deformedCageHandle == InvalidHandle)
	{
		return;
	}

	// Processes the current actions and selects elements from the mesh.
	UpdateMeshSelection(viewInfo);

	// We are okay to update all proxies now since they have been already marked dirty.
	_scene->UpdateDirtyRenderProxies();
}

void Editor::UpdateMeshSelection(const ViewInfo& viewInfo)
{
	if (_inputSubsystem->GetMouseButtonsState() == MouseButtonsState::LeftPressed ||
		_isDragging ||
		_isSelectingRect)
	{
		return;
	}

	const auto deformedCageMesh = _scene->GetMesh(_deformedCageHandle);
	const auto mousePosition = _inputSubsystem->GetMousePosition();

	// If we have the gizmo selected or highlighted we don't want to highlight any mesh elements.
	if (_isGizmoHighlighted || _isGizmoTransformed)
	{
		if (_statusBar->GetActiveSelectionType() == SelectionType::Vertex)
		{
			auto selection = deformedCageMesh->GetSelection<SelectionType::Vertex>();
			selection.UnhighlightAll();
		}
		else if (_statusBar->GetActiveSelectionType() == SelectionType::Edge)
		{
			auto selection = deformedCageMesh->GetSelection<SelectionType::Edge>();
			selection.UnhighlightAll();
		}
		else if (_statusBar->GetActiveSelectionType() == SelectionType::Polygon)
		{
			auto selection = deformedCageMesh->GetSelection<SelectionType::Polygon>();
			selection.UnhighlightAll();
		}

		return;
	}

	if (_statusBar->GetActiveSelectionType() == SelectionType::Vertex)
	{
		// If we are doing vertex selection first cache the projected vertices into screen space.
		deformedCageMesh->MarkCachedGeometryDirty();
		deformedCageMesh->CacheProjectedPointsWorldToScreen(viewInfo);

		auto selection = deformedCageMesh->GetSelection<SelectionType::Vertex>();
		const auto hit = deformedCageMesh->QueryClosestPointScreenSpace(viewInfo,
			mousePosition,
			VertexSelectionMinDistanceThresholdSq);

		if (hit.has_value())
		{
			if (_highlightedVertexHandle != hit->_vertexHandle && _highlightedVertexHandle.is_valid())
			{
				selection.Unhighlight(_highlightedVertexHandle);
			}

			selection.Highlight(hit->_vertexHandle);

			_highlightedVertexHandle = hit->_vertexHandle;
		}
		else
		{
			if (_highlightedVertexHandle.is_valid())
			{
				selection.Unhighlight(_highlightedVertexHandle);
			}

			_highlightedVertexHandle = VertexHandle();
		}
	}
	else if (_statusBar->GetActiveSelectionType() == SelectionType::Edge)
	{
		// If we are doing vertex selection first cache the projected vertices into screen space.
		deformedCageMesh->MarkCachedGeometryDirty();
		deformedCageMesh->CacheProjectedPointsWorldToScreen(viewInfo);

		auto selection = deformedCageMesh->GetSelection<SelectionType::Edge>();
		const auto hit = deformedCageMesh->QueryClosestEdgeScreenSpace(viewInfo,
			mousePosition,
			EdgeSelectionMinDistanceThreshold);

		if (hit.has_value())
		{
			if (_highlightedEdgeHandle != hit->_edgeHandle && _highlightedEdgeHandle.is_valid())
			{
				selection.Unhighlight(_highlightedEdgeHandle);
			}

			selection.Highlight(hit->_edgeHandle);

			_highlightedEdgeHandle = hit->_edgeHandle;
		}
		else
		{
			if (_highlightedEdgeHandle.is_valid())
			{
				selection.Unhighlight(_highlightedEdgeHandle);
			}

			_highlightedEdgeHandle = EdgeHandle();
		}
	}
	else if (_statusBar->GetActiveSelectionType() == SelectionType::Polygon)
	{
		auto selection = deformedCageMesh->GetSelection<SelectionType::Polygon>();
		const auto worldRay = viewInfo.DeprojectScreenToWorldRay(mousePosition);
		const auto hit = deformedCageMesh->QueryRayHit(worldRay);

		if (hit._polyHandle.is_valid())
		{
			if (_highlightedPolygonHandle != hit._polyHandle && _highlightedPolygonHandle.is_valid())
			{
				selection.Unhighlight(_highlightedPolygonHandle);
			}

			selection.Highlight(hit._polyHandle);

			_highlightedPolygonHandle = hit._polyHandle;
		}
		else
		{
			selection.UnhighlightAll();

			_highlightedPolygonHandle = FaceHandle();
		}
	}
}

void Editor::SetUpUIElements()
{
	// Create both pointers in the project model.
	_projectModel = std::make_shared<ProjectModelData>();

	_statusBar = std::make_shared<StatusBar>(_meshOperationSystem);
	_projectOptionsPanel = std::make_shared<ProjectOptionsPanel>(_projectModel,
		_meshOperationSystem,
		[this](const bool shouldCompute) { OnComputeInfluenceColorMap(shouldCompute); },
		[this] { OnNewProjectCreated(); });
	_toolBar = std::make_shared<ToolBar>(_inputSubsystem, _meshOperationSystem, _toolSystem);
}

void Editor::OnNewProjectCancelled()
{
	_newProjectPanel = nullptr;
}

void Editor::OnProjectSettingsApplied()
{
	// Manually update the model here on the project options panel, otherwise the temporary "modified" state will not update.
	_projectOptionsPanel->SetModelData(_projectModel);

	// Re-create the project since we changed the data model.
	OnNewProjectCreated();
}

void Editor::OnNewProjectCreated(const std::shared_ptr<std::promise<void>>& completionPromise)
{
	LOG_DEBUG("Set Cage and Mesh");

	_projectCreationFailed.store(false, std::memory_order_seq_cst);

	if (_projectModel->CheckMissingFiles())
	{
		_statusBar->SetError("Unable to load all files, check if some of them are missing.");
		_projectCreationFailed.store(true, std::memory_order_seq_cst);
		return;
	}

	_isComputingWeightsData.store(true, std::memory_order_seq_cst);
	_isComputingDeformationData.store(false, std::memory_order_seq_cst);

	auto projectModelSnapshot = std::make_shared<ProjectModelData>(*_projectModel);
	if (_newProjectPanel != nullptr)
	{
		projectModelSnapshot = std::make_shared<ProjectModelData>(*_newProjectPanel->GetModel());
	}
	projectModelSnapshot->ApplyPMVCPreset();

	const bool isEvaluationMode = _isEvaluationMode;

	_threadPool->Submit([this, completionPromise, projectModelSnapshot, isEvaluationMode]()
	{
		const auto initStart = std::chrono::steady_clock::now();
		const auto fail = [this]()
		{
			_isComputingWeightsData.store(false, std::memory_order_seq_cst);
			_isComputingDeformationData.store(false, std::memory_order_seq_cst);
			_projectCreationFailed.store(true, std::memory_order_seq_cst);
		};

		auto projectResult = _meshOperationSystem->ExecuteOperation<MeshLoadOperation>(
			projectModelSnapshot->_deformationType,
			projectModelSnapshot->_LBCWeightingScheme,
			projectModelSnapshot->_meshFilepath.value(),
			projectModelSnapshot->_cageFilepath.value(),
			projectModelSnapshot->_deformedCageFilepath,
			projectModelSnapshot->_weightsFilepath,
			projectModelSnapshot->_embeddingFilepath,
			projectModelSnapshot->_parametersFilepath,
			projectModelSnapshot->_numBBWSteps,
			projectModelSnapshot->_numSamples,
			projectModelSnapshot->_scalingFactor,
			projectModelSnapshot->_interpolateWeights,
			projectModelSnapshot->_findOffset,
			projectModelSnapshot->_noOffset,
			projectModelSnapshot->_pmvcUseOffset,
			projectModelSnapshot->_somigNu,
			projectModelSnapshot->_somiglianaDeformer);

		if (projectResult.HasError())
		{
			_mainThreadQueue->Push([this, error = std::move(projectResult.GetError())]() mutable
			{
				_statusBar->SetError(std::move(error));
			});

			fail();
			return;
		}

		auto projectData = projectResult.GetValue();
		const auto initEnd = std::chrono::steady_clock::now();
		const auto initMs = std::chrono::duration<double, std::milli>(initEnd - initStart).count();

		using WeightsResult = decltype(ComputeCageWeights(*projectData));
		std::future<WeightsResult> future;

		if (DeformationTypeHelpers::IsPMVC(projectModelSnapshot->_deformationType))
		{
			auto promise = std::make_shared<std::promise<WeightsResult>>();
			future = promise->get_future();

			_mainThreadQueue->Push([this, projectData, projectModelSnapshot, promise]() mutable
			{
				try
				{
					_cubemapRenderer->Initialize(PMVCSettings::kCubemapSize);
					_cubemapRenderer->SetCage(projectData->_cage);
					_cubemapRenderer->SetMesh(projectData->_mesh);
					promise->set_value(_cubemapRenderer->ComputeCoordinates(
						projectData->_pmvcUseOffset,
						static_cast<uint32_t>(projectModelSnapshot->_pmvcHitCount),
						projectModelSnapshot->_pmvcSkipEvenHits,
						projectModelSnapshot->_pmvcUseInteriorDistance));
				}
				catch (...)
				{
					promise->set_exception(std::current_exception());
				}
			});
		}
		
		else
		{
			std::promise<WeightsResult> promise;
			future = promise.get_future();
			promise.set_value(ComputeCageWeights(*projectData));
		}

		auto weightsResult = future.get();
		/*try
		{
			weightsResult = future.get();
		}
		catch (const std::exception& e)
		{
			_mainThreadQueue->Push([this, msg = std::string(e.what())]() mutable
			{
				_statusBar->SetError(std::move(msg));
			});

			fail();
			return;
		}
		catch (...)
		{
			_mainThreadQueue->Push([this]()
			{
				_statusBar->SetError("Unknown error during main-thread weight computation.");
			});

			fail();
			return;
		}*/

		if (weightsResult.HasError())
		{
			_mainThreadQueue->Push([this, error = std::move(weightsResult.GetError())]() mutable
			{
				_statusBar->SetError(std::move(error));
			});

			fail();
			return;
		}

		_isComputingDeformationData.store(true, std::memory_order_seq_cst);
		_isComputingWeightsData.store(false, std::memory_order_seq_cst);

		const auto renderMs = weightsResult.GetValue()._renderMs;
		const auto computeMs = weightsResult.GetValue()._computeMs;
		const auto computeTotalMs = weightsResult.GetValue()._computeTotalMs;
		const auto transferMs = weightsResult.GetValue()._transferMs;
		const auto cubemapInitMs = weightsResult.GetValue()._initMs;

		_weightsData.Update(std::move(weightsResult.GetValue()._skinningMatrix),
			std::move(weightsResult.GetValue()._weights),
			std::move(weightsResult.GetValue()._interpolatedWeights),
			std::move(weightsResult.GetValue()._psi),
			std::move(weightsResult.GetValue()._psiTri),
			std::move(weightsResult.GetValue()._psiQuad));

		const auto& mesh = projectData->_mesh;
		const auto& cage = projectData->_cage;
		const auto& defCage = projectData->_deformedCage;

		LOG_DEBUG("MESH vertices: {} x {}", mesh._vertices.rows(), mesh._vertices.cols());
		LOG_DEBUG("CAGE vertices: {} x {}", cage._vertices.rows(), cage._vertices.cols());
		LOG_DEBUG("DEF CAGE vertices: {} x {}", defCage._vertices.rows(), defCage._vertices.cols());

		const auto deformationApplyStart = std::chrono::steady_clock::now();
		auto deformedMeshResult = ComputeDeformedMesh(projectData->_mesh,
			projectData->_cage,
			projectData->_deformedCage,
			projectData->_deformationType,
			projectData->_pmvcUseOffset,
			projectData->_LBCWeightingScheme,
			projectData->_somiglianaDeformer,
			projectData->_modelVerticesOffset,
			projectData->_numSamples,
			projectData->CanInterpolateWeights());

		if (deformedMeshResult.HasError())
		{
			_mainThreadQueue->Push([this, error = std::move(deformedMeshResult.GetError())]() mutable
			{
				_statusBar->SetError(std::move(error));
			});

			fail();
			return;
		}

		_deformationData.Update(std::move(deformedMeshResult.GetValue()._vertexData));
		const auto deformationApplyEnd = std::chrono::steady_clock::now();
		const auto deformationApplyMs = std::chrono::duration<double, std::milli>(deformationApplyEnd - deformationApplyStart).count();

		if (isEvaluationMode)
		{
			// Evaluation/offscreen mode:
			// keep computed data, skip all scene/render proxy churn.
			_projectData = projectData;
			{
				std::scoped_lock lock(_evaluationTimingsMutex);
				_latestEvaluationStageTimings._initMs = initMs + cubemapInitMs.value_or(0.0);
				_latestEvaluationStageTimings._computeTotalMs = computeTotalMs;
				if (DeformationTypeHelpers::IsPMVC(projectData->_deformationType))
				{
					_latestEvaluationStageTimings._renderMs = renderMs;
					_latestEvaluationStageTimings._computeMs = computeMs;
					_latestEvaluationStageTimings._transferMs = transferMs;
				}
				else
				{
					_latestEvaluationStageTimings._renderMs.reset();
					_latestEvaluationStageTimings._computeMs.reset();
					_latestEvaluationStageTimings._transferMs.reset();
				}
				_latestEvaluationStageTimings._deformationApplyMs = deformationApplyMs;
			}

			_isComputingDeformationData.store(false, std::memory_order_seq_cst);

			if (completionPromise != nullptr)
			{
				try
				{
					completionPromise->set_value();
				}
				catch (const std::future_error&)
				{
				}
			}

			return;
		}

		_mainThreadQueue->Push([this, projectData, completionPromise]() mutable
		{
			const auto& viewInfo = _cameraSubsystem->GetCamera().GetViewInfo();
			_gizmo->SetPosition(viewInfo, glm::vec3(0.0f));

			if (_deformedMeshHandle != InvalidHandle)
			{
				_scene->RemoveMesh(_deformedMeshHandle);
				_deformedMeshHandle = InvalidHandle;
			}

			if (_deformedCageHandle != InvalidHandle)
			{
				_scene->RemoveMesh(_deformedCageHandle);
				_deformedCageHandle = InvalidHandle;
			}

			_projectData = projectData;

			const auto translation = glm::translate(glm::mat4(1.0f), glm::vec3(_projectData->_centerOffset));
			const auto scale = glm::scale(glm::mat4(1.0f), glm::vec3(_projectData->_scalingFactor));
			const auto newModelMatrix = scale * translation;

			_deformedMeshHandle = _scene->AddMesh(_projectData->_mesh._vertices, _projectData->_mesh._faces);
			const auto deformedMesh = _scene->GetMesh(_deformedMeshHandle);
			deformedMesh->SetModelMatrix(newModelMatrix);

			_deformedCageHandle = _scene->AddCage(_projectData->_deformedCage._vertices, _projectData->_deformedCage._faces);
			const auto cageMesh = _scene->GetMesh(_deformedCageHandle);
			cageMesh->SetModelMatrix(newModelMatrix);

			const auto renderInfluenceMap = _projectModel->CanRenderInfluenceMap();
			if (renderInfluenceMap)
			{
				UpdateMeshVertexColors(renderInfluenceMap);
			}

			_projectOptionsPanel->SetDeformableMesh(_scene->GetMesh(_deformedMeshHandle));
			_projectOptionsPanel->SetCageMesh(_scene->GetMesh(_deformedCageHandle));

			_toolBar->SetModel(std::make_shared<ToolBarModel>(_projectData));

			_statusBar->SetModel(std::make_shared<StatusBarModel>(_projectData,
				[this]<typename T>(T && selectionType) { OnSelectionTypeChanged(std::forward<T>(selectionType)); },
				[this](const uint32_t newFrameIndex) { OnSequencerFrameIndexChanged(newFrameIndex); },
				[this](const uint32_t frameIndex, const uint32_t numFrames) { OnSequencerNumFramesChanged(frameIndex, numFrames); },
				[this]() { OnSequencerStartedDragging(); },
				[this]() { OnSequencerEndedDragging(); }));

			UpdateDeformedMeshPositionsFromDeformationData();

			{
				auto bvhBuilder = _scene->BeginGeometryBVH();
				bvhBuilder.AddGeometry(_deformedMeshHandle);
				bvhBuilder.AddGeometry(_deformedCageHandle);
			}

			if (_newProjectPanel != nullptr)
			{
				_newProjectPanel->Dismiss();
				_newProjectPanel = nullptr;
			}

			if (_projectSettingsPanel != nullptr)
			{
				_projectSettingsPanel->Dismiss();
				_projectSettingsPanel = nullptr;
			}

			_isComputingDeformationData.store(false, std::memory_order_seq_cst);

			if (completionPromise != nullptr)
			{
				try
				{
					completionPromise->set_value();
				}
				catch (const std::future_error&)
				{
				}
			}
		});
	});
}

void Editor::ClearEvaluationData()
{
	/* {
		auto emptyWeights = MeshComputeWeightsOperationResult{};
		_weightsData.Update(std::move(emptyWeights._skinningMatrix),
			std::move(emptyWeights._weights),
			std::move(emptyWeights._interpolatedWeights),
			std::move(emptyWeights._psi),
			std::move(emptyWeights._psiTri),
			std::move(emptyWeights._psiQuad));
	}



	_weightsData.Update(Eigen::MatrixXd(),
		Eigen::MatrixXd(),
		Eigen::MatrixXd(),
		Eigen::MatrixXd(),
		Eigen::MatrixXd(),
		Eigen::MatrixXd());
	*/
	//_deformationData.Update({});

	_projectData.reset();


}

void Editor::OnProjectSettingsCancelled()
{
	_projectSettingsPanel = nullptr;
}

void Editor::UpdateGizmoSelection(const ViewInfo& viewInfo, const GizmoType activeGizmoType)
{
	const auto mousePosition = _inputSubsystem->GetMousePosition();
	const auto closestHit = _gizmo->QueryRayHit(viewInfo, activeGizmoType, mousePosition);

	if (closestHit.has_value())
	{
		if (!_isGizmoHighlighted || _highlightedGizmoAxis != closestHit->_axis)
		{
			_gizmo->SetHighlighted(activeGizmoType, _highlightedGizmoAxis, false);

			_highlightedGizmoAxis = closestHit->_axis;
			_isGizmoHighlighted = true;

			_gizmo->SetHighlighted(activeGizmoType, closestHit->_axis, true);
		}
	}
	else
	{
		// If we didn't hit anything, but we also have had something highlighted, then unhighlight.
		if (_isGizmoHighlighted && !_isGizmoTransformed)
		{
			_isGizmoHighlighted = false;

			_gizmo->SetHighlighted(activeGizmoType, _highlightedGizmoAxis, false);
		}
	}
}

void Editor::OnClicked(const InputActionParams& actionParams)
{
	if ((_newProjectPanel != nullptr && _newProjectPanel->IsModalPanelVisible()) ||
		(_projectSettingsPanel != nullptr && _projectSettingsPanel->IsModalPanelVisible()) ||
		ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow))
	{
		return;
	}

	// If we don't have a valid mesh.
	if (_deformedMeshHandle == InvalidHandle || _deformedCageHandle == InvalidHandle)
	{
		return;
	}

	// If we are computing the weights of a new mesh we don't want to be able to do anything until we have it ready.
	if (_isComputingWeightsData.load(std::memory_order_relaxed))
	{
		return;
	}

	if (actionParams._keyState == KeyState::KeyDown)
	{
		OnMouseClickPressed(actionParams);
	}
	else if (actionParams._keyState == KeyState::KeyPressed)
	{
		OnMouseMoved(actionParams);
	}
	else if (actionParams._keyState == KeyState::KeyUp)
	{
		OnMouseClickReleased(actionParams);
	}
}

void Editor::OnMouseClickPressed(const InputActionParams& actionParams)
{
	// Update the camera focus point if we have hit the cage or any other object in the scene.
	// const auto currentMouseRay = viewInfo.DeprojectScreenToWorldRay(mousePosition);
	// const auto meshHitResult = _scene->QueryClosestMesh(currentMouseRay);
	//
	// 	camera.SetPointOfInterest(meshHitResult->_worldPosition);

	const auto& camera = _cameraSubsystem->GetCamera();
	const auto& viewInfo = camera.GetViewInfo();
	const auto prevMousePosition = _inputSubsystem->GetPreviousMousePosition();
	const auto mousePosition = _inputSubsystem->GetMousePosition();

	if (_isGizmoHighlighted)
	{
		_isGizmoTransformed = true;

		_activeMeshTransformation = MeshTransformation(viewInfo,
			_scene->GetMesh(_deformedCageHandle),
			_statusBar->GetActiveSelectionType(),
			GetTransformationTypeFromGizmoType(_activeGizmoType),
			GetTransformationAxisFromGizmoAxis(_highlightedGizmoAxis),
			_gizmo->GetMatrix(),
			mousePosition,
			prevMousePosition);

		// We bail early here otherwise we will end up selecting a vertex.
		return;
	}

	const auto hasValidHighlight = _highlightedVertexHandle.is_valid() || _highlightedEdgeHandle.is_valid() || _highlightedPolygonHandle.is_valid();
	const auto hasActiveSelection = hasValidHighlight || _isGizmoHighlighted || _isDragging;
	const auto modifierKeys = _inputSubsystem->GetKeyModifiers();
	const auto hasModifierKeys = IsSet(modifierKeys, SDL_KMOD_LSHIFT) || IsSet(modifierKeys, SDL_KMOD_LALT) || IsSet(modifierKeys, SDL_KMOD_LGUI);

	// If we are making a rectangle selection on the screen we want to do it before we process any mesh selection, so we can use the data.
	if (!hasActiveSelection && !hasModifierKeys)
	{
		_selectionRectStartPosition = ImVec2(mousePosition.x, mousePosition.y);
		_selectionRectEndPosition = _selectionRectStartPosition;
		_isSelectingRect = true;
	}
}

void Editor::OnMouseMoved(const InputActionParams& actionParams)
{
	const auto &camera = _cameraSubsystem->GetCamera();
	const auto &viewInfo = camera.GetViewInfo();
	const auto prevMousePosition = _inputSubsystem->GetPreviousMousePosition();
	const auto mousePosition = _inputSubsystem->GetMousePosition();

	// If we are doing a rectangle selection skip anything else.
	if (_isSelectingRect)
	{
		_selectionRectEndPosition = ImVec2(mousePosition.x, mousePosition.y);

		return;
	}

	// If we have dragged the mouse then we mark it as dragged until mouse up event.
	const auto mouseDelta = _inputSubsystem->GetMousePosition() - _inputSubsystem->GetPreviousMousePosition();
	_isDragging |= (_isGizmoHighlighted && (glm::length2(mouseDelta) > Epsilon));
	_hasDragged |= _isDragging;

	if (_activeMeshTransformation.has_value() && _isDragging && !HasModifierKeysPressed(actionParams._modifierKeys))
	{
		// First transform the gizmo based on the mouse input.
		_activeMeshTransformation->Transform(viewInfo,
			mousePosition,
			prevMousePosition);

		ResetGizmoPositionFromSelection(viewInfo);
	}
}

void Editor::OnMouseClickReleased(const InputActionParams& actionParams)
{
	_activeMeshTransformation.reset();

	const auto& camera = _cameraSubsystem->GetCamera();
	const auto& viewInfo = camera.GetViewInfo();
	const auto prevMousePosition = _inputSubsystem->GetPreviousMousePosition();
	const auto mousePosition = _inputSubsystem->GetMousePosition();

	// If we are doing a rectangle selection skip anything else.
	if (_isSelectingRect)
	{
		// If we are doing selection first cache the projected vertices into screen space.
		const auto deformedCageMesh = _scene->GetMesh(_deformedCageHandle);
		deformedCageMesh->CacheProjectedPointsWorldToScreen(viewInfo);

		const auto selectionType = _statusBar->GetActiveSelectionType();
		const glm::vec2 rectMin(std::min(_selectionRectStartPosition.x, _selectionRectEndPosition.x), std::min(_selectionRectStartPosition.y, _selectionRectEndPosition.y));
		const glm::vec2 rectMax(std::max(_selectionRectStartPosition.x, _selectionRectEndPosition.x), std::max(_selectionRectStartPosition.y, _selectionRectEndPosition.y));

		if (selectionType == SelectionType::Vertex)
		{
			auto selection = deformedCageMesh->GetSelection<SelectionType::Vertex>();
			selection.SelectInRectangle(viewInfo, rectMin, rectMax);
		}
		else if (selectionType == SelectionType::Edge)
		{
			auto selection = deformedCageMesh->GetSelection<SelectionType::Edge>();
			selection.SelectInRectangle(viewInfo, rectMin, rectMax);
		}
		else if (selectionType == SelectionType::Polygon)
		{
			auto selection = deformedCageMesh->GetSelection<SelectionType::Polygon>();
			selection.SelectInRectangle(viewInfo, rectMin, rectMax);
		}

		_isSelectingRect = false;

		ResetGizmoPositionFromSelection(viewInfo);

		return;
	}

	if (_isGizmoTransformed && _hasDragged && !HasModifierKeysPressed(actionParams._modifierKeys))
	{
		if (_deformedMeshHandle == InvalidHandle || _deformedCageHandle == InvalidHandle)
		{
			return;
		}

		const bool useOriginalMeshForOffsetPMVC = _projectData->_pmvcUseOffset &&
			DeformationTypeHelpers::IsPMVC(_projectData->_deformationType);
		auto meshForRecompute = useOriginalMeshForOffsetPMVC
			? _projectData->_mesh
			: _scene->GetMesh(_deformedMeshHandle)->CopyAsEigen();

		// Re-compute the deformed mesh and update the render proxy.
		_threadPool->Submit([this,
			mesh = std::move(meshForRecompute),
			cage = _projectData->_cage,
			deformedMesh = _scene->GetMesh(_deformedCageHandle)->CopyAsEigen(),
			somiglianaDeformer = _projectData->_somiglianaDeformer,
			deformationType = _projectData->_deformationType,
			pmvcUseOffset = _projectData->_pmvcUseOffset,
			weightingScheme = _projectData->_LBCWeightingScheme,
			modelVerticesOffset = _projectData->_modelVerticesOffset,
			numSamples = _projectData->_numSamples,
			interpolateWeights = _projectData->CanInterpolateWeights()]() mutable
		{
			_isComputingDeformationData.store(true, std::memory_order_seq_cst);

			auto deformedMeshResult = ComputeDeformedMesh(std::move(mesh),
				std::move(cage),
				std::move(deformedMesh),
				deformationType,
				pmvcUseOffset,
				weightingScheme,
				somiglianaDeformer,
				modelVerticesOffset,
				numSamples,
				interpolateWeights);

			_deformationData.Update(std::move(deformedMeshResult.GetValue()._vertexData));

			_isComputingDeformationData.store(false, std::memory_order_seq_cst);

			// Copy the matrix to transpose in-place, so we can iterate over it in the correct memory data layout.
			_mainThreadQueue->Push([this]() mutable
			{
				// Update the positions of the mesh.
				UpdateDeformedMeshPositionsFromDeformationData();

				// We only recompute the vertex colors if they were previously on.
				if (_projectModel->_renderInfluenceMap)
				{
					UpdateMeshVertexColors(_projectModel->_renderInfluenceMap);
				}

				// Rebuild the entire BVH.
				{
					auto bvhBuilder = _scene->BeginGeometryBVH();
					bvhBuilder.AddGeometry(_deformedMeshHandle);
					bvhBuilder.AddGeometry(_deformedCageHandle);
				}
			});
		});
	}

	OnClickedSelection(actionParams);

	_isDragging = false;
	_hasDragged = false;
	_isGizmoTransformed = false;
}

void Editor::OnClickedSelection(const InputActionParams& actionParams)
{
	if (_deformedMeshHandle == InvalidHandle || _deformedCageHandle == InvalidHandle)
	{
		return;
	}

	const auto& viewInfo = _cameraSubsystem->GetCamera().GetViewInfo();
	const auto deformedCageMesh = _scene->GetMesh(_deformedCageHandle);
	const auto selectionType = _statusBar->GetActiveSelectionType();

	if (!_isDragging && !_hasDragged)
	{
		// If we are doing selection first cache the projected vertices into screen space.
		deformedCageMesh->CacheProjectedPointsWorldToScreen(viewInfo);

		const auto mousePosition = _inputSubsystem->GetMousePosition();

		if (selectionType == SelectionType::Vertex)
		{
			auto selection = deformedCageMesh->GetSelection<SelectionType::Vertex>();
			auto hit = deformedCageMesh->QueryClosestPointScreenSpace(viewInfo,
				mousePosition,
				VertexSelectionMinDistanceThresholdSq);

			if (hit.has_value())
			{
				const auto handles = std::span(&hit->_vertexHandle, 1);

				if (IsSet(actionParams._modifierKeys, SDL_KMOD_LALT))
				{
					selection.Deselect(handles);
				}
				else if (IsSet(actionParams._modifierKeys, SDL_KMOD_LSHIFT))
				{
					selection.Select(handles);
				}
				else
				{
					selection.DeselectAll();
					selection.Select(handles);
				}
			}
		}
		else if (selectionType == SelectionType::Edge)
		{
			auto selection = deformedCageMesh->GetSelection<SelectionType::Edge>();
			auto hit = deformedCageMesh->QueryClosestEdgeScreenSpace(viewInfo, mousePosition, EdgeSelectionMinDistanceThreshold);

			if (hit.has_value())
			{
				const auto handles = std::span(&hit->_edgeHandle, 1);

				if (IsSet(actionParams._modifierKeys, SDL_KMOD_LALT))
				{
					selection.Deselect(handles);
				}
				else if (IsSet(actionParams._modifierKeys, SDL_KMOD_LSHIFT))
				{
					selection.Select(handles);
				}
				else
				{
					selection.DeselectAll();
					selection.Select(handles);
				}
			}
		}
		else if (selectionType == SelectionType::Polygon)
		{
			auto selection = deformedCageMesh->GetSelection<SelectionType::Polygon>();
			const auto worldRay = viewInfo.DeprojectScreenToWorldRay(mousePosition);
			auto hit = deformedCageMesh->QueryRayHit(worldRay);

			if (hit._polyHandle.is_valid())
			{
				const auto handles = std::span(&hit._polyHandle, 1);

				if (IsSet(actionParams._modifierKeys, SDL_KMOD_LALT))
				{
					selection.Deselect(handles);
				}
				else if (IsSet(actionParams._modifierKeys, SDL_KMOD_LSHIFT))
				{
					selection.Select(handles);
				}
				else
				{
					selection.DeselectAll();
					selection.Select(handles);
				}
			}
		}

		ResetGizmoPositionFromSelection(viewInfo);
	}
}

void Editor::OnToolSelectionChanged(const ToolType toolType)
{
	const auto activeGizmoType = _toolBar->GetActiveGizmoType();

	// Set the visibility of the gizmos based on the selected tool from the tool bar.
	if (_activeGizmoType != activeGizmoType)
	{
		_activeGizmoType = activeGizmoType;

		for (std::size_t i = 0; i < static_cast<std::size_t>(GizmoType::MaxNum); ++i)
		{
			const auto gizmoType = static_cast<GizmoType>(i);
			const auto isVisible = (gizmoType == activeGizmoType);

			_gizmo->SetVisible(gizmoType, isVisible);
		}
	}

	if (toolType == Tools::Transform::Translate)
	{
		const auto& viewInfo = _cameraSubsystem->GetCamera().GetViewInfo();

		ResetGizmoPositionFromSelection(viewInfo);
	}
}

void Editor::OnSelectionTypeChanged(const SelectionType selectionType)
{
	const auto& viewInfo = _cameraSubsystem->GetCamera().GetViewInfo();
	const auto deformedCageMesh = _scene->GetMesh(_deformedCageHandle);
	deformedCageMesh->CacheProjectedPointsWorldToScreen(viewInfo);

	// If we are doing vertex selection first cache the projected vertices into screen space.
	if (selectionType == SelectionType::Vertex)
	{
		deformedCageMesh->SetWireframeRenderMode(WireframeRenderMode::Points | WireframeRenderMode::Edges);
	}
	else if (selectionType == SelectionType::Edge)
	{
		deformedCageMesh->SetWireframeRenderMode(WireframeRenderMode::Edges);
	}
	else if (selectionType == SelectionType::Polygon)
	{
		deformedCageMesh->SetWireframeRenderMode(WireframeRenderMode::Edges | WireframeRenderMode::Polygons);
	}
	else
	{
		deformedCageMesh->SetWireframeRenderMode(WireframeRenderMode::None);
	}

	ResetGizmoPositionFromSelection(viewInfo);
}

void Editor::OnSequencerFrameIndexChanged(const uint32_t newFrameIndex)
{
	if (_deformedMeshHandle == InvalidHandle || _deformedCageHandle == InvalidHandle)
	{
		return;
	}

	// Update deformed mesh vertices with the new sample index data.
	UpdateDeformedMeshPositionsFromDeformationData(newFrameIndex);

	// We only recompute the vertex colors if they were previously on.
	if (_projectModel->_renderInfluenceMap)
	{
		UpdateMeshVertexColors(_projectModel->_renderInfluenceMap);
	}
}

void Editor::OnSequencerNumFramesChanged(const uint32_t currentFrameIndex, const uint32_t numFrames)
{
	if (_deformedMeshHandle == InvalidHandle || _deformedCageHandle == InvalidHandle)
	{
		return;
	}

	const bool useOriginalMeshForOffsetPMVC = _projectData->_pmvcUseOffset &&
		DeformationTypeHelpers::IsPMVC(_projectData->_deformationType);
	auto meshForRecompute = useOriginalMeshForOffsetPMVC
		? _projectData->_mesh
		: _scene->GetMesh(_deformedMeshHandle)->CopyAsEigen();

	// Re-compute the deformed mesh and update the render proxy.
	_threadPool->Submit([this,
		mesh = std::move(meshForRecompute),
		cage = _projectData->_cage,
		deformedMesh = _scene->GetMesh(_deformedCageHandle)->CopyAsEigen(),
		somiglianaDeformer = _projectData->_somiglianaDeformer,
		deformationType = _projectData->_deformationType,
		pmvcUseOffset = _projectData->_pmvcUseOffset,
		weightingScheme = _projectData->_LBCWeightingScheme,
		modelVerticesOffset = _projectData->_modelVerticesOffset,
		numSamples = _projectData->_numSamples,
		interpolateWeights = _projectData->CanInterpolateWeights(),
		currentFrameIndex]() mutable
	{
		_isComputingDeformationData.store(true, std::memory_order_seq_cst);

		auto deformedMeshResult = ComputeDeformedMesh(std::move(mesh),
			std::move(cage),
			std::move(deformedMesh),
			deformationType,
			pmvcUseOffset,
			weightingScheme,
			somiglianaDeformer,
			modelVerticesOffset,
			numSamples,
			interpolateWeights);

		_deformationData.Update(std::move(deformedMeshResult.GetValue()._vertexData));

		_isComputingDeformationData.store(false, std::memory_order_seq_cst);

		// Copy the matrix to transpose in-place, so we can iterate over it in the correct memory data layout.
		_mainThreadQueue->Push([this, currentFrameIndex]() mutable
		{
			// Update the positions of the mesh.
			UpdateDeformedMeshPositionsFromDeformationData(currentFrameIndex);

			// We only recompute the vertex colors if they were previously on.
			if (_projectModel->_renderInfluenceMap)
			{
				UpdateMeshVertexColors(_projectModel->_renderInfluenceMap);
			}
		});
	});
}

void Editor::OnSequencerStartedDragging()
{
	const auto deformedCageMesh = _scene->GetMesh(_deformedCageHandle);
	deformedCageMesh->SetVisible(false);
}

void Editor::OnSequencerEndedDragging()
{
	const auto deformedCageMesh = _scene->GetMesh(_deformedCageHandle);
	deformedCageMesh->SetVisible(true);
}

void Editor::UpdateDeformedMeshPositionsFromDeformationData(const std::optional<uint32_t> frameIndex)
{
	// Copy the matrix to transpose in-place, so we can iterate over it in the correct memory data layout.
	const auto deformationData = _deformationData.LockRead();
	const auto unpackedFrameIndex = frameIndex.value_or(deformationData->_vertexData.size() - 1);
	const auto meshPositions = deformationData->_vertexData[unpackedFrameIndex]._vertices.transpose();
	std::vector<glm::vec3> positions = GeometryUtils::EigenVerticesToGLM(meshPositions);

	const auto deformedMesh = _scene->GetMesh(_deformedMeshHandle);
	deformedMesh->SetPositions(positions);

	_statusBar->SetCurrentFrame(unpackedFrameIndex);
}

void Editor::ExportCurrentDeformedMesh(std::filesystem::path filepath) const
{
	CheckFormat(!_isComputingDeformationData.load(std::memory_order_relaxed), "The weights and the deformation mesh haven't been computed yet to export.");

	const auto deformationData = _deformationData.LockRead();

	_meshOperationSystem->ExecuteOperation<DeformedMeshExportOperation>(
		*deformationData,
		_projectData->_deformationType,
		_projectData->_LBCWeightingScheme,
		_statusBar->GetCurrentFrameIndex(),
		_projectData->_somiglianaDeformer,
		_projectData->_mesh._faces,
		std::move(filepath),
		1.0f / _projectData->_scalingFactor);
}

void Editor::ExportDeformedMeshes(std::filesystem::path filepath) const
{
	CheckFormat(!_isComputingDeformationData.load(std::memory_order_relaxed), "The weights and the deformation mesh haven't been computed yet to export.");

	const auto deformationData = _deformationData.LockRead();

	_meshOperationSystem->ExecuteOperation<DeformedMeshExportOperation>(
		*deformationData,
		_projectData->_deformationType,
		_projectData->_LBCWeightingScheme,
		std::optional<std::size_t>(),
		_projectData->_somiglianaDeformer,
		_projectData->_mesh._faces,
		std::move(filepath),
		1.0f / _projectData->_scalingFactor);
}

void Editor::ExportDeformedCage(std::filesystem::path filepath) const
{
	_meshOperationSystem->ExecuteOperation<MeshExportOperation>(
		_projectData->_deformationType,
		_projectData->_LBCWeightingScheme,
		_projectData->_deformedCage._faces,
		_projectData->_deformedCage._vertices,
		std::move(filepath));
}

void Editor::ExportInfluenceColorMap(std::filesystem::path filepath,
	std::optional<std::vector<int32_t>> selectedVertices) const
{
	CheckFormat(!_isComputingWeightsData.load(std::memory_order_relaxed), "The weights and the deformation mesh haven't been computed yet to export.");
	if (!selectedVertices.has_value() && !_projectData->_parametrization.has_value())
	{
		LOG_WARN("Skipping influence map export because parametrization data is missing and no selected vertices were provided.");
		return;
	}

	auto parametrization = _projectData->_parametrization.value_or(Parametrization { });
	auto weights = *_weightsData.LockRead();

	_meshOperationSystem->ExecuteOperation<MeshExportInfluenceMapOperation>(
		_projectData->_deformationType,
		_projectData->_LBCWeightingScheme,
		_projectData->_somiglianaDeformer,
		_projectData->_mesh,
		_projectData->_cage,
		std::move(parametrization),
		std::move(selectedVertices),
		std::move(filepath),
		std::move(weights),
		_projectData->_modelVerticesOffset,
		_projectData->CanInterpolateWeights());
}

void Editor::ExportDistanceFieldColorMap(std::filesystem::path filepath,
	const bool useEuclideanDistance,
	std::optional<std::vector<int32_t>> selectedVertices,
	DistanceColorMapParams colorMapParams) const
{
	CheckFormat(!_isComputingWeightsData.load(std::memory_order_relaxed), "The weights and the deformation mesh haven't been computed yet to export.");
	if (!selectedVertices.has_value() && !_projectData->_parametrization.has_value())
	{
		LOG_WARN("Skipping distance field export because parametrization data is missing and no selected vertices were provided.");

		return;
	}

	// Only the interior distances have to be read from somewhere, the Euclidean ones follow
	// from the vertex positions alone.
	const Eigen::MatrixXf* interiorDetours = nullptr;
	if (!useEuclideanDistance)
	{
		// The interior distances are read back from the table the interior distance PMVC
		// variant filled, an empty table means the field does not exist yet.
		interiorDetours = (_cubemapRenderer != nullptr) ? &_cubemapRenderer->GetInteriorDetours() : nullptr;

		if (interiorDetours == nullptr || interiorDetours->size() == 0)
		{
			LOG_WARN("Skipping interior distance field export because no interior distances have been computed for this project. "
				"Run the project with the interior distance PMVC variant first, or export the euclidean distance field instead.");

			return;
		}
	}

	auto parametrization = _projectData->_parametrization.value_or(Parametrization { });

	_meshOperationSystem->ExecuteOperation<MeshExportDistanceFieldOperation>(
		_projectData->_mesh,
		_projectData->_cage,
		std::move(parametrization),
		std::move(selectedVertices),
		std::move(filepath),
		interiorDetours,
		std::move(colorMapParams));
}

void Editor::OnComputeInfluenceColorMap(const bool shouldRenderInfluenceMap) const
{
	UpdateMeshVertexColors(shouldRenderInfluenceMap && _projectModel->CanRenderInfluenceMap());
}

void Editor::ExportWeights(std::filesystem::path filepath) const
{
	CheckFormat(!_isComputingWeightsData.load(std::memory_order_relaxed), "The weights and the deformation mesh haven't been computed yet to export.");
	const auto weightsData = _weightsData.LockRead();
	const auto weightsToExport = DeformationTypeHelpers::IsPMVC(_projectData->_deformationType)
		? weightsData->_weights.transpose()
		: weightsData->_weights;

	//const auto weightsData = _weightsData.LockRead();
	const auto embedding = _projectData->_embedding.value_or(EigenMesh{ });

	_meshOperationSystem->ExecuteOperation<MeshExportWeightsOperation>(
		_projectData->_deformationType,
		_projectData->_LBCWeightingScheme,
		std::move(filepath),
		std::move(weightsToExport),
		_projectData->_b,
		_projectData->_bc,
		embedding,
		_projectData->_numBBWSteps);
}

MeshOperationResult<std::shared_ptr<ProjectData>> Editor::CreateProject() const
{
	return _meshOperationSystem->ExecuteOperation<MeshLoadOperation>(
		_projectModel->_deformationType,
		_projectModel->_LBCWeightingScheme,
		_projectModel->_meshFilepath.value(),
		_projectModel->_cageFilepath.value(),
		_projectModel->_deformedCageFilepath,
		_projectModel->_weightsFilepath,
		_projectModel->_embeddingFilepath,
		_projectModel->_parametersFilepath,
		_projectModel->_numBBWSteps,
		_projectModel->_numSamples,
		_projectModel->_scalingFactor,
		_projectModel->_interpolateWeights,
		_projectModel->_findOffset,
		_projectModel->_noOffset,
		_projectModel->_pmvcUseOffset,
		_projectModel->_somigNu,
		_projectModel->_somiglianaDeformer
		);
}

MeshOperationResult<MeshComputeWeightsOperationResult> Editor::ComputeCageWeights(const ProjectData& projectData) const
{
	return _meshOperationSystem->ExecuteOperation<MeshComputeWeightsOperation>(
		projectData._deformationType,
		projectData._LBCWeightingScheme,
		projectData._mesh,
		projectData._cage,
		projectData._embedding,
		projectData._weights,
		projectData._somiglianaDeformer,
		projectData._cagePoints,
		projectData._normals,
		projectData._b,
		projectData._bc,
		projectData.CanInterpolateWeights(),
		projectData._numBBWSteps,
		projectData._numSamples);
}

MeshOperationResult<MeshComputeDeformationOperationResult> Editor::ComputeDeformedMesh(EigenMesh mesh,
	EigenMesh cage,
	EigenMesh deformedCage,
	const DeformationType deformationType,
	const bool pmvcUseOffset,
	const LBC::DataSetup::WeightingScheme weightingScheme,
	const std::shared_ptr<somig_deformer_3>& somiglianaDeformer,
	const int32_t modelVerticesOffset,
	const int32_t numSamples,
	const bool interpolateWeights) const
{
	auto weights = *_weightsData.LockRead();

	return _meshOperationSystem->ExecuteOperation<MeshComputeDeformationOperation>(
		deformationType,
		pmvcUseOffset,
		weightingScheme,
		somiglianaDeformer,
		std::move(mesh),
		std::move(cage),
		std::move(deformedCage),
		std::move(weights),
		modelVerticesOffset,
		numSamples,
		interpolateWeights);
}

void Editor::ResetGizmoPositionFromSelection(const ViewInfo& viewInfo) const
{
	const auto activeSelectionType = _statusBar->GetActiveSelectionType();
	const auto cageMesh = _scene->GetMesh(_deformedCageHandle);

	const auto updateGizmoVisibility = [this](const auto& selection)
	{
		if (selection.HasSelection())
		{
			const auto activeGizmoType = _toolBar->GetActiveGizmoType();

			for (std::size_t i = 0; i < static_cast<std::size_t>(GizmoType::MaxNum); ++i)
			{
				const auto gizmoType = static_cast<GizmoType>(i);
				const auto isVisible = (gizmoType == activeGizmoType);

				_gizmo->SetVisible(gizmoType, isVisible);
			}
		}
		else
		{
			// Hide the gizmo if we have no selection.
			_gizmo->SetVisible(false);
		}
	};

	glm::vec3 averageSelPosition(0.0f);

	if (activeSelectionType == SelectionType::Vertex)
	{
		// Get the current vertex selection and apply the average position to the gizmo.
		const auto selection = cageMesh->GetSelection<SelectionType::Vertex>();
		const auto vertexSelection = selection.GetSelection();

		updateGizmoVisibility(selection);

		for (const auto vertexHandle : vertexSelection)
		{
			averageSelPosition += cageMesh->GetAverageVertexPosition(vertexHandle);
		}

		averageSelPosition /= std::max(vertexSelection.size(), 1_sz);
	}
	else if (activeSelectionType == SelectionType::Edge)
	{
		// Get the current vertex selection and apply the average position to the gizmo.
		const auto selection = cageMesh->GetSelection<SelectionType::Edge>();
		const auto uniqueVertices = selection.GetVertexSelection();

		updateGizmoVisibility(selection);

		for (const auto vertexHandle : uniqueVertices)
		{
			averageSelPosition += cageMesh->GetAverageVertexPosition(vertexHandle);
		}

		averageSelPosition /= std::max(uniqueVertices.size(), 1_sz);
	}
	else if (activeSelectionType == SelectionType::Polygon)
	{
		// Get the current vertex selection and apply the average position to the gizmo.
		const auto selection = cageMesh->GetSelection<SelectionType::Polygon>();
		const auto faceSelection = selection.GetSelection();

		updateGizmoVisibility(selection);

		for (const auto faceHandle : faceSelection)
		{
			averageSelPosition += cageMesh->GetAverageVertexPosition(faceHandle);
		}

		averageSelPosition /= std::max(faceSelection.size(), 1_sz);
	}

	// Update the gizmo position.
	_gizmo->SetPosition(viewInfo, averageSelPosition);
}

void Editor::UpdateMeshVertexColors(const bool shouldRenderInfluenceMap) const
{
	const auto deformedMesh = _scene->GetMesh(_deformedMeshHandle);

	if (shouldRenderInfluenceMap)
	{
		CheckFormat(!_isComputingWeightsData.load(std::memory_order_relaxed), "The weights and the deformation mesh haven't been computed yet to export.");



		if (!_projectData->_parametrization.has_value())
		{
			LOG_WARN("Skipping influence map color rendering because parametrization data is missing.");
			//deformedMesh->SetColors(std::vector<glm::vec3>(deformedMesh->GetNumVertices(), glm::vec3(0.0f)), false);
			return;
		}


		auto weights = *_weightsData.LockRead();

		const auto vertexColorsResult = _meshOperationSystem->ExecuteOperation<MeshComputeInfluenceMapOperation>(
			_projectData->_deformationType,
			_projectData->_LBCWeightingScheme,
			_projectData->_somiglianaDeformer,
			_projectData->_mesh._vertices,
			*_projectData->_parametrization,
			std::move(weights),
			_projectData->_modelVerticesOffset,
			_projectData->CanInterpolateWeights());

		CheckFormat(!vertexColorsResult.HasError(), "The vertex colors should not fail.");

		const auto colors = vertexColorsResult.GetValue()._vertexColors.transpose();

		std::vector<glm::vec3> vertexColors(colors.cols());

		for (auto i = 0; i < colors.cols(); ++i)
		{
			vertexColors[i] = glm::vec3(colors(0, i), colors(1, i), colors(2, i));
		}

		deformedMesh->SetColors(vertexColors, true);
	}
	else
	{
		std::vector<glm::vec3> vertexColors(deformedMesh->GetNumVertices());

		for (auto& vertexColor : vertexColors)
		{
			vertexColor = glm::vec3(0.0f);
		}

		deformedMesh->SetColors(vertexColors, false);
	}
}
