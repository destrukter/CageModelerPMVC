#version 450

layout(location = 0) in vec3 OutColor;
layout(location = 1) flat in float TriangleID;

layout(location = 0) out vec4 FragColor;

void main()
{
    if (gl_FragCoord.z <= 0.0 || gl_FragCoord.z >= 0.999999)
    {
        discard;
    }

    float id = float(TriangleID);
    FragColor = vec4(OutColor, id);
}