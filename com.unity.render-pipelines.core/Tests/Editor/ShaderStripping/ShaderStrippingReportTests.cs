using System;
using UnityEditor;
using UnityEditor.Build;
using NUnit.Framework;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace UnityEditor.Rendering.Tests
{
    public class ShaderStrippingReportTest
    {
        class BuildReportTestScope : IDisposable
        {
            private IPreprocessBuildWithReport m_PreProcessReport;
            private IPostprocessBuildWithReport m_PostProcessReport;

            public BuildReportTestScope()
            {
                var instance = Activator.CreateInstance(Type.GetType("UnityEditor.Rendering.ShaderStrippingReportScope, Unity.RenderPipelines.Core.Editor"));
                m_PostProcessReport = instance as IPostprocessBuildWithReport;
                m_PreProcessReport = instance as IPreprocessBuildWithReport;
                m_PreProcessReport.OnPreprocessBuild(default);
            }

            void IDisposable.Dispose()
            {
                m_PostProcessReport.OnPostprocessBuild(default);
            }
        }


        [Test]
        public void CheckReportIsCorrect()
        {
            using (new BuildReportTestScope())
            {
                var shaders = new List<Shader>() { Shader.Find("UI/Default"), Shader.Find("Sprites/Default") };
                foreach (var shader in shaders)
                {
                    for (uint i = 0; i < 5; ++i)
                    {
                        uint variantsIn = 10 * i;
                        ShaderStrippingReport.instance.OnShaderProcessed<Shader, ShaderSnippetData>(shader, default, variantsIn, (uint)(variantsIn * 0.5), i);
                    }
                }
            }
        }
    }
}
