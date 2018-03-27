#ifndef LIGHTWEIGHT_INPUT_SURFACE_UNLIT_INCLUDED
#define LIGHTWEIGHT_INPUT_SURFACE_UNLIT_INCLUDED

#include "Core.hlsl"

CBUFFER_START(UnityPerMaterial)
float4 _MainTex_ST;
half4 _Color;
half _Cutoff;
half _Glossiness;
half _Metallic;
CBUFFER_END

#include "InputSurfaceCommon.hlsl"

#endif // LIGHTWEIGHT_INPUT_SURFACE_UNLIT_INCLUDED
