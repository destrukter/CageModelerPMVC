#include <Mesh/Operations/MeshComputeInfluenceMapOperation.h>

#include <cagedeformations/InfluenceMap.h>

MeshComputeInfluenceMapOperation::ExecutionResult MeshComputeInfluenceMapOperation::Execute()
{
	const auto controlVerticesIdx = ResolveControlVertexIndices(std::nullopt, _params._parametrization);
	const auto weights = ResolveInfluenceWeights(_params._deformationType,
		_params._weightsData.get(),
		_params._somiglianaDeformer,
		_params._vertices.rows(),
		_params._modelVerticesOffset,
		_params._interpolateWeights);

	return MeshComputeInfluenceMapOperationResult { influence_color_map(AccumulateInfluences(weights, controlVerticesIdx)) };
}
