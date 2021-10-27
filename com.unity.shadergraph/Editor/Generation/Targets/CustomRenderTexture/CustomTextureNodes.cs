using UnityEngine;
using UnityEditor.ShaderGraph;
using UnityEditor.Graphing;
using UnityEditor.ShaderGraph.Internal;

namespace UnityEditor.Rendering.CustomRenderTexture.ShaderGraph
{
    [Title("Custom Render Texture", "Size")]
    [SubTargetFilter(typeof(CustomTextureSubTarget))]
    class CustomTextureSize : AbstractMaterialNode, IGeneratesFunction
    {
        private const string kOutputSlotWidthName = "Texture Width";
        private const string kOutputSlotHeightName = "Texture Height";
        private const string kOutputSlotDepthName = "Texture Depth";

        public const int OutputSlotWidthId = 0;
        public const int OutputSlotHeightId = 1;
        public const int OutputSlotDepthId = 2;

        public CustomTextureSize()
        {
            name = "Custom Render Texture Size";
            UpdateNodeAfterDeserialization();
        }

        protected int[] validSlots => new[] { OutputSlotWidthId, OutputSlotHeightId, OutputSlotDepthId };

        public sealed override void UpdateNodeAfterDeserialization()
        {
            AddSlot(new Vector1MaterialSlot(OutputSlotWidthId, kOutputSlotWidthName, kOutputSlotWidthName, SlotType.Output, 0));
            AddSlot(new Vector1MaterialSlot(OutputSlotHeightId, kOutputSlotHeightName, kOutputSlotHeightName, SlotType.Output, 0));
            AddSlot(new Vector1MaterialSlot(OutputSlotDepthId, kOutputSlotDepthName, kOutputSlotDepthName, SlotType.Output, 0));
            RemoveSlotsNameNotMatching(validSlots);
        }

        public override string GetVariableNameForSlot(int slotId)
        {
            switch (slotId)
            {
                case OutputSlotHeightId:
                    return "_CustomRenderTextureHeight";
                case OutputSlotDepthId:
                    return "_CustomRenderTextureDepth";
                default:
                    return "_CustomRenderTextureWidth";
            }
        }

        public void GenerateNodeFunction(FunctionRegistry registry, GenerationMode generationMode)
        {
            // For preview only we declare CRT defines
            if (generationMode == GenerationMode.Preview)
            {
                registry.builder.AppendLine("#define _CustomRenderTextureHeight 1.0");
                registry.builder.AppendLine("#define _CustomRenderTextureWidth 1.0");
                registry.builder.AppendLine("#define _CustomRenderTextureDepth 1.0");
            }
        }
    }

    [Title("Custom Render Texture", "Slice Index / Cubemap Face")]
    [SubTargetFilter(typeof(CustomTextureSubTarget))]
    class CustomTextureSlice : AbstractMaterialNode, IGeneratesFunction
    {
        private const string kOutputSlotCubeFaceName = "Texture Cube Face";
        private const string kOutputSlot3DSliceName = "Texture 3D Slice";

        public const int OutputSlotCubeFaceId = 3;
        public const int OutputSlot3DSliceId = 4;

        public CustomTextureSlice()
        {
            name = "Slice Index / Cubemap Face";
            UpdateNodeAfterDeserialization();
        }

        protected int[] validSlots => new[] { OutputSlotCubeFaceId, OutputSlot3DSliceId };

        public sealed override void UpdateNodeAfterDeserialization()
        {
            AddSlot(new Vector1MaterialSlot(OutputSlotCubeFaceId, kOutputSlotCubeFaceName, kOutputSlotCubeFaceName, SlotType.Output, 0));
            AddSlot(new Vector1MaterialSlot(OutputSlot3DSliceId, kOutputSlot3DSliceName, kOutputSlot3DSliceName, SlotType.Output, 0));
            RemoveSlotsNameNotMatching(validSlots);
        }

        public override string GetVariableNameForSlot(int slotId)
        {
            switch (slotId)
            {
                case OutputSlotCubeFaceId:
                    return "_CustomRenderTextureCubeFace";
                case OutputSlot3DSliceId:
                    return "_CustomRenderTexture3DSlice";
                default:
                    return "_CustomRenderTextureWidth";
            }
        }

        public void GenerateNodeFunction(FunctionRegistry registry, GenerationMode generationMode)
        {
            // For preview only we declare CRT defines
            if (generationMode == GenerationMode.Preview)
            {
                registry.builder.AppendLine("#define _CustomRenderTextureCubeFace 0.0");
                registry.builder.AppendLine("#define _CustomRenderTexture3DSlice 0.0");
            }
        }
    }

    [Title("Custom Render Texture", "Self")]
    [SubTargetFilter(typeof(CustomTextureSubTarget))]

    class CustomTextureSelf : AbstractMaterialNode, IGeneratesFunction
    {
        private const string kOutputSlotSelf2DName = "Self Texture 2D";
        private const string kOutputSlotSelfCubeName = "Self Texture Cube";
        private const string kOutputSlotSelf3DName = "Self Texture 3D";

        public const int OutputSlotSelf2DId = 5;
        public const int OutputSlotSelfCubeId = 6;
        public const int OutputSlotSelf3DId = 7;

        public CustomTextureSelf()
        {
            name = "Custom Render Texture Self";
            UpdateNodeAfterDeserialization();
        }

        protected int[] validSlots => new[] { OutputSlotSelf2DId, OutputSlotSelfCubeId, OutputSlotSelf3DId };

        public sealed override void UpdateNodeAfterDeserialization()
        {
            AddSlot(new Texture2DMaterialSlot(OutputSlotSelf2DId, kOutputSlotSelf2DName, kOutputSlotSelf2DName, SlotType.Output, ShaderStageCapability.Fragment, false) { bareResource = true });
            AddSlot(new CubemapMaterialSlot(OutputSlotSelfCubeId, kOutputSlotSelfCubeName, kOutputSlotSelfCubeName, SlotType.Output, ShaderStageCapability.Fragment, false) { bareResource = true });
            AddSlot(new Texture3DMaterialSlot(OutputSlotSelf3DId, kOutputSlotSelf3DName, kOutputSlotSelf3DName, SlotType.Output, ShaderStageCapability.Fragment, false) { bareResource = true });
            RemoveSlotsNameNotMatching(validSlots);
        }

        public override string GetVariableNameForSlot(int slotId)
        {
            switch (slotId)
            {
                case OutputSlotSelf2DId:
                    return "UnityBuildTexture2DStructNoScale(_SelfTexture2D)";
                case OutputSlotSelfCubeId:
                    return "UnityBuildTextureCubeStruct(_SelfTextureCube)";
                default:
                    return "UnityBuildTexture3DStruct(_SelfTexture3D)";
            }
        }

        public void GenerateNodeFunction(FunctionRegistry registry, GenerationMode generationMode)
        {
            // For preview only we declare CRT defines
            if (generationMode == GenerationMode.Preview)
            {
                registry.builder.AppendLine("#if !defined(UNITY_CRT_PREVIEW_TEXTURE) && !defined(UNITY_CUSTOM_TEXTURE_INCLUDED)");
                registry.builder.AppendLine("#define UNITY_CRT_PREVIEW_TEXTURE");
                registry.builder.AppendLine("TEXTURE2D(_SelfTexture2D);");
                registry.builder.AppendLine("SAMPLER(sampler_SelfTexture2D);");
                registry.builder.AppendLine("float4 _SelfTexture2D_TexelSize;");
                registry.builder.AppendLine("TEXTURECUBE(_SelfTextureCube);");
                registry.builder.AppendLine("SAMPLER(sampler_SelfTextureCube);");
                registry.builder.AppendLine("float4 _SelfTextureCube_TexelSize;");
                registry.builder.AppendLine("TEXTURE3D(_SelfTexture3D);");
                registry.builder.AppendLine("SAMPLER(sampler_SelfTexture3D);");
                registry.builder.AppendLine("float4 sampler_SelfTexture3D_TexelSize;");
                registry.builder.AppendLine("#endif");
            }
        }
    }

    [Title("Custom Render Texture", "Current Dimension")]
    [SubTargetFilter(typeof(CustomTextureSubTarget))]

    class CustomTextureDimension : AbstractMaterialNode, IGeneratesBodyCode
    {
        private const string kOutputSlot2D = "Is 2D";
        private const string kOutputSlot3D = "Is 3D";
        private const string kOutputSlotCube = "Is Cube";

        public const int kOutputSlot2DId = 0;
        public const int kOutputSlot3DId = 1;
        public const int kOutputSlotCubeId = 2;

        public CustomTextureDimension()
        {
            name = "Custom Render Texture Dimension";
            UpdateNodeAfterDeserialization();
        }

        protected int[] validSlots => new[] { kOutputSlot2DId, kOutputSlot3DId, kOutputSlotCubeId };

        public sealed override void UpdateNodeAfterDeserialization()
        {
            AddSlot(new BooleanMaterialSlot(kOutputSlot2DId, kOutputSlot2D, kOutputSlot2D, SlotType.Output, false));
            AddSlot(new BooleanMaterialSlot(kOutputSlot3DId, kOutputSlot3D, kOutputSlot3D, SlotType.Output, false));
            AddSlot(new BooleanMaterialSlot(kOutputSlotCubeId, kOutputSlotCube, kOutputSlotCube, SlotType.Output, false));
            RemoveSlotsNameNotMatching(validSlots);
        }

        public void GenerateNodeCode(ShaderStringBuilder sb, GenerationMode generationMode)
        {
            if (generationMode.IsPreview())
            {
                sb.AppendLine(@"bool {0} = false;", GetVariableNameForSlot(kOutputSlot2DId));
                sb.AppendLine(@"bool {0} = false;", GetVariableNameForSlot(kOutputSlot3DId));
                sb.AppendLine(@"bool {0} = false;", GetVariableNameForSlot(kOutputSlotCubeId));
            }
            else
            {
                sb.AppendLine(@"bool {0} = CustomRenderTextureDimension == CRT_DIMENSION_2D;", GetVariableNameForSlot(kOutputSlot2DId));
                sb.AppendLine(@"bool {0} = CustomRenderTextureDimension == CRT_DIMENSION_3D;", GetVariableNameForSlot(kOutputSlot3DId));
                sb.AppendLine(@"bool {0} = CustomRenderTextureDimension == CRT_DIMENSION_CUBE;", GetVariableNameForSlot(kOutputSlotCubeId));
            }
        }
    }
}
