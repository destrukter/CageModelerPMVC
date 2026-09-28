#include <Mesh/Operations/MeshExportInfluenceMapOperation.h>

#include <cagedeformations/InfluenceMap.h>

void MeshExportInfluenceMapOperation::Execute()
{
	const auto controlVerticesIdx = ResolveControlVertexIndices(_params._selectedVertices, _params._parametrization);
	const auto weights = ResolveInfluenceWeights(_params._deformationType,
		_params._weightsData.get(),
		_params._somiglianaDeformer,
		_params._mesh._vertices.rows(),
		_params._modelVerticesOffset,
		_params._interpolateWeights);

	write_influence_color_map_OBJ(_params._outputFilepath.string(),
		_params._mesh._vertices,
		_params._mesh._faces,
		AccumulateInfluences(weights, controlVerticesIdx));
}
