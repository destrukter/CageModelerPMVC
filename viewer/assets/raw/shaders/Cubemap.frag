#version 450

layout(location = 0) in vec4 OutColor;
layout(location = 1) flat in float PrimitiveID;

layout(location = 0) out vec4 FragColor;
layout(location = 1) out vec4 FragDepthHistory;

void main()
{
    FragColor = OutColor;
    FragDepthHistory = vec4(gl_FragCoord.z, 0.0, 0.0, PrimitiveID);
}
