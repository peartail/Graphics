using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using System.Linq;
using System.Text;
using JetBrains.Annotations;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEngine;
using UnityEngine.Rendering;

namespace UnityEditor.Rendering
{
    [Serializable]
    class VariantCounter
    {
        public uint inputVariants;
        public uint outputVariants;
        public override string ToString() => $"Total={inputVariants}/{outputVariants}({outputVariants / (float)inputVariants * 100f:0.00}%)";
    }

    [Serializable]
    class ShaderVariantInfo : VariantCounter
    {
        public string variantName;
        public double stripTimeMs;
        public override string ToString() => $"{variantName} - {base.ToString()} - Time={stripTimeMs}ms";
    }

    [Serializable]
    class ShaderStrippingInfo : VariantCounter, ISerializationCallbackReceiver
    {
        public string name;

        private Dictionary<string, List<ShaderVariantInfo>> m_VariantsByPipeline = new();

        public void AddVariant(string pipeline, ShaderVariantInfo variant)
        {
            if (!m_VariantsByPipeline.TryGetValue(pipeline, out var list))
            {
                list = new List<ShaderVariantInfo>();
                m_VariantsByPipeline.Add(pipeline, list);
            }

            inputVariants += variant.inputVariants;
            outputVariants += variant.outputVariants;

            list.Add(variant);
        }

        public override string ToString() => $"{name} - {base.ToString()}";

        public void Log(ShaderVariantLogLevel logLevel)
        {
            IEnumerable<ShaderVariantInfo> variantsToLog = null;
            switch (logLevel)
            {
                case ShaderVariantLogLevel.AllShaders:
                {
                    variantsToLog = m_VariantsByPipeline.SelectMany(i => i.Value);
                    break;
                }
                case ShaderVariantLogLevel.OnlySRPShaders:
                {
                    variantsToLog = m_VariantsByPipeline.
                        Where(i => !string.IsNullOrEmpty(i.Key))
                        .SelectMany(i => i.Value);
                        break;
                }
            }

            if (variantsToLog != null && variantsToLog.Any())
            {
                StringBuilder sb = new StringBuilder();
                sb.AppendLine($"{this}");
                foreach (var info in m_VariantsByPipeline)
                {
                    foreach (var variant in info.Value)
                    {
                        sb.AppendLine($"- {variant}");
                    }
                }

                Debug.Log(sb.ToString());
            }
        }

        #region Serialization & Export

        [Serializable]
        class PipelineVariants
        {
            public string pipeline;
            public ShaderVariantInfo[] variants;
        }

        [SerializeField] private PipelineVariants[] pipelines;
        public void OnBeforeSerialize()
        {
            pipelines = m_VariantsByPipeline
                .Select(pipeline => new PipelineVariants() { pipeline = pipeline.Key, variants = pipeline.Value.ToArray() })
                .ToArray();
        }

        public void OnAfterDeserialize()
        {
            pipelines = null;
        }

        #endregion
    }

    /// <summary>
    /// This class works as an scope of the <see cref="ShaderStrippingReport"/> hooking into the
    /// <see cref="IPreprocessBuildWithReport"/> that are being called at the begin of the build and
    /// to <see cref="IPostprocessBuildWithReport"/> that are the ones called after the build is finished
    /// </summary>
    class ShaderStrippingReportScope : IPostprocessBuildWithReport, IPreprocessBuildWithReport
    {
        internal static bool s_DefaultExport = false;
        /// <summary>
        /// Callback order
        /// </summary>
        public int callbackOrder => 0;

        /// <summary>
        /// Creates the Report
        /// </summary>
        /// <param name="report">Unused <see cref="BuildReport"/></param>
        public void OnPreprocessBuild(BuildReport report)
        {
            ShaderStrippingReport.instance = new();
        }

        /// <summary>
        /// Dumps the report into the console and export the variants
        /// </summary>
        /// <param name="report">Unused <see cref="BuildReport"/></param>
        public void OnPostprocessBuild(BuildReport report)
        {
            ShaderVariantLogLevel logStrippedVariants = ShaderVariantLogLevel.Disabled;
            bool exportStrippedVariants = s_DefaultExport;

            // Check the current pipeline and check the shader variant settings
            if (RenderPipelineManager.currentPipeline != null && RenderPipelineManager.currentPipeline.defaultSettings is IShaderVariantSettings shaderVariantSettings)
            {
                logStrippedVariants = shaderVariantSettings.shaderVariantLogLevel;
                exportStrippedVariants = shaderVariantSettings.exportShaderVariants;
            }

            // Dump everything into console and/or file
            ShaderStrippingReport.instance.Dump(logStrippedVariants, exportStrippedVariants);

            // Release resources
            ShaderStrippingReport.instance = null;
        }
    }

    /// <summary>
    /// Class to gather all the information about stripping in SRP packages
    /// </summary>
    public class ShaderStrippingReport
    {
        // Shader
        private readonly List<ShaderStrippingInfo> m_ShaderInfos = new();
        private readonly VariantCounter m_ShaderVariantCounter = new();

        // Compute Shader
        private readonly List<ShaderStrippingInfo> m_ComputeShaderInfos = new();
        private readonly VariantCounter m_ComputeShaderVariantCounter = new();

        #region Public API

        /// <summary>
        /// Instance to the current build report for shader stripping
        /// </summary>
        public static ShaderStrippingReport instance { get; internal set; }

        /// <summary>
        /// Method to report a variant stripping
        /// </summary>
        /// <param name="shader">Shader can either be <see cref="Shader"/> or <see cref="ComputeShader"/></param>
        /// <param name="shaderVariant">A variant for the given shader</param>
        /// <param name="variantsIn">The input variants that were given to the stripper</param>
        /// <param name="variantsOut">The output variants that were given to the stripper</param>
        /// <param name="stripTimeMs">The total stripping time in milliseconds</param>
        /// <typeparam name="TShader">The type of the shader</typeparam>
        /// <typeparam name="TShaderVariant">The type of the variant</typeparam>
        /// <exception cref="NotImplementedException">The exception if the Shader doesn't match it's supported variant</exception>
        public void OnShaderProcessed<TShader, TShaderVariant>([DisallowNull] TShader shader, TShaderVariant shaderVariant, uint variantsIn, uint variantsOut, double stripTimeMs)
            where TShader : UnityEngine.Object
        {
            if (!TryGetPipelineAndVariantName(shader, shaderVariant, out string pipeline, out string variantName))
                throw new NotImplementedException($"Report is not enabled for {typeof(TShader)} and {typeof(TShaderVariant)}");

            var lastShaderStrippingInfo = FindLastShaderStrippingInfo(shader);
            if (typeof(TShader) == typeof(Shader))
            {
                m_ShaderVariantCounter.inputVariants += variantsIn;
                m_ShaderVariantCounter.outputVariants += variantsOut;
            }
            else if (typeof(TShader) == typeof(ComputeShader))
            {
                m_ComputeShaderVariantCounter.inputVariants += variantsIn;
                m_ComputeShaderVariantCounter.outputVariants += variantsOut;
            }

            lastShaderStrippingInfo.AddVariant(pipeline, new ShaderVariantInfo()
            {
                inputVariants = variantsIn,
                outputVariants = variantsOut,
                stripTimeMs = stripTimeMs,
                variantName = variantName
            });
        }

        #endregion

        #region Private API
        [CanBeNull] private ShaderStrippingInfo m_LastShaderStrippingInfo = null;

        private ShaderStrippingInfo FindLastShaderStrippingInfo<TShader>([DisallowNull] TShader shader)
            where TShader : UnityEngine.Object
        {
            if (m_LastShaderStrippingInfo != null && m_LastShaderStrippingInfo.name.Equals(shader.name))
                return m_LastShaderStrippingInfo;

            // We are reporting a new shader variant, need to create a new one
            m_LastShaderStrippingInfo = new ShaderStrippingInfo()
            {
                name = shader.name
            };

            // The compiler will strip the branch that we are not using
            if (typeof(TShader) == typeof(Shader))
            {
                m_ShaderInfos.Add(m_LastShaderStrippingInfo);
            }
            else if (typeof(TShader) == typeof(ComputeShader))
            {
                m_ComputeShaderInfos.Add(m_LastShaderStrippingInfo);
            }

            return m_LastShaderStrippingInfo;
        }

        [MustUseReturnValue]
        private static bool TryGetPipelineAndVariantName<TShader, TShaderVariant>([DisallowNull] TShader shader, TShaderVariant shaderVariant, out string pipeline, out string variantName)
            where TShader : UnityEngine.Object
        {
            pipeline = string.Empty;
            variantName = string.Empty;

            if (typeof(TShader) == typeof(Shader) && typeof(TShaderVariant) == typeof(ShaderSnippetData))
            {
                var inputShader = (Shader)Convert.ChangeType(shader, typeof(Shader));
                var snippetData = (ShaderSnippetData)Convert.ChangeType(shaderVariant, typeof(ShaderSnippetData));
                pipeline = inputShader.GetRenderPipelineTag(snippetData);
                string passName = string.IsNullOrEmpty(snippetData.passName)
                    ? $"Pass {snippetData.pass.PassIndex}"
                    : snippetData.passName;
                variantName = $"{passName} ({snippetData.passType}) (SubShader: {snippetData.pass.SubshaderIndex})";
            }
            else if (typeof(TShader) == typeof(ComputeShader) && typeof(TShaderVariant) == typeof(string))
            {
                variantName = $"Kernel: {shaderVariant}";
            }
            else
            {
                return false;
            }

            return true;
        }

        #endregion



        #region Serialization & Export

        [Serializable]
        class Export
        {
            public uint totalVariantsIn;
            public uint totalVariantsOut;
            public ShaderStrippingInfo[] shaders;
        }

        void ExportShaderStrippingInfo(string path, VariantCounter variantCounter, List<ShaderStrippingInfo> shaders)
        {
            try
            {
                var export = new Export()
                {
                    totalVariantsIn = variantCounter.inputVariants,
                    totalVariantsOut = variantCounter.outputVariants,
                    shaders = shaders.ToArray()
                };

                File.WriteAllText(path, JsonUtility.ToJson(export, true));
            }
            catch (Exception e)
            {
                Debug.LogException(e);
            }
        }

        internal void Dump(ShaderVariantLogLevel shaderVariantLogLevel, bool exportStrippedVariants)
        {
            if (shaderVariantLogLevel != ShaderVariantLogLevel.Disabled)
            {
                Debug.Log($"Shader Stripping - {m_ShaderVariantCounter}");
                foreach (var info in m_ShaderInfos)
                {
                    info.Log(shaderVariantLogLevel);
                }

                Debug.Log($"Compute Shader Stripping - {m_ComputeShaderVariantCounter}");
                foreach (var info in m_ComputeShaderInfos)
                {
                    info.Log(shaderVariantLogLevel);
                }
            }

            if (exportStrippedVariants)
            {
                ExportShaderStrippingInfo("Temp/shader-stripping.json", m_ShaderVariantCounter, m_ShaderInfos);
                ExportShaderStrippingInfo("Temp/compute-shader-stripping.json", m_ComputeShaderVariantCounter, m_ComputeShaderInfos);
            }
        }

        #endregion
    }
}
