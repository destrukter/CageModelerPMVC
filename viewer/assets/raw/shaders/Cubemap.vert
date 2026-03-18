#version 450

layout(push_constant) uniform PushConstants {
    mat4 view;
    mat4 proj;
    int faceIndex;
} pc;

layout(location = 0) in vec3 InPosition;
layout(location = 1) in uint InPrimitiveID;
layout(location = 2) in uint InVertexIndex;

layout(location = 0) out vec4 OutColor;
layout(location = 1) flat out float PrimitiveID;

layout(set = 0, binding = 0) uniform FrameDataBlock
{
    float invNumPrimitives;
} ubo;

void main()
{
    vec4 color = vec4(0.0);
    if (InVertexIndex == 0u) color = vec4(1.0, 0.0, 0.0, 0.0);
    else if (InVertexIndex == 1u) color = vec4(0.0, 1.0, 0.0, 0.0);
    else if (InVertexIndex == 2u) color = vec4(0.0, 0.0, 1.0, 0.0);
    else color = vec4(0.0, 0.0, 0.0, 1.0);

    OutColor = color;
    PrimitiveID = float(InPrimitiveID) * ubo.invNumPrimitives;

    gl_Position = pc.proj * pc.view * vec4(InPosition, 1.0);
}
