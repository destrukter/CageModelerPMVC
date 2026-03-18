#version 450

layout(location = 0) in vec4 OutColor;
layout(location = 1) flat in float PrimitiveID;

layout(set = 1, binding = 0) uniform sampler2DArray uPrevDepth;

layout(push_constant) uniform PushConstants {
    mat4 view;
    mat4 proj;
    int faceIndex;
} pc;

layout(location = 0) out vec4 FragColor;
layout(location = 1) out vec4 FragDepthHistory;

void main()
{
    ivec2 px = ivec2(gl_FragCoord.xy);
    float previousDepth = texelFetch(uPrevDepth, ivec3(px, pc.faceIndex), 0).r;

    if (gl_FragCoord.z <= previousDepth)
    {
        discard;
    }

    FragColor = OutColor;
    FragDepthHistory = vec4(gl_FragCoord.z, 0.0, 0.0, PrimitiveID);
}
