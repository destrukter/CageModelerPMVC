#version 450

layout(set = 1, binding = 0) uniform sampler2D PrevDepth;

layout(location = 0) in vec3 OutColor;
layout(location = 1) flat in float TriangleID;

layout(location = 0) out vec4 FragColor;

void main()
{
    float z = gl_FragCoord.z;

    if (z <= 0.0 || z >= 0.999999)
    {
        discard;
    }

    ivec2 coord = ivec2(gl_FragCoord.xy);
    float prevDepth = texelFetch(PrevDepth, coord, 0).r;

    const float eps = 1e-6;

    // Peel: keep only fragments behind the previous hit
    if (z <= prevDepth + eps)
    {
        discard;
    }

    float id = float(TriangleID);
    FragColor = vec4(OutColor, id);
}