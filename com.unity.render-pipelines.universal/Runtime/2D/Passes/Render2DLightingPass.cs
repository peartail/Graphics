using System.Collections.Generic;
using UnityEngine.Profiling;
using UnityEngine.U2D;

namespace UnityEngine.Rendering.Universal
{
    internal class Render2DLightingPass : ScriptableRenderPass, IRenderPass2D
    {
        private static readonly int k_HDREmulationScaleID = Shader.PropertyToID("_HDREmulationScale");
        private static readonly int k_InverseHDREmulationScaleID = Shader.PropertyToID("_InverseHDREmulationScale");
        private static readonly int k_UseSceneLightingID = Shader.PropertyToID("_UseSceneLighting");
        private static readonly int k_RendererColorID = Shader.PropertyToID("_RendererColor");
        private static readonly int k_CameraSortingLayerTextureID = Shader.PropertyToID("_CameraSortingLayerTexture");

        private static readonly int[] k_ShapeLightTextureIDs =
        {
            Shader.PropertyToID("_ShapeLightTexture0"),
            Shader.PropertyToID("_ShapeLightTexture1"),
            Shader.PropertyToID("_ShapeLightTexture2"),
            Shader.PropertyToID("_ShapeLightTexture3")
        };

        private static readonly ShaderTagId k_CombinedRenderingPassName = new ShaderTagId("Universal2D");
        private static readonly ShaderTagId k_NormalsRenderingPassName = new ShaderTagId("NormalsRendering");
        private static readonly ShaderTagId k_LegacyPassName = new ShaderTagId("SRPDefaultUnlit");
        private static readonly List<ShaderTagId> k_ShaderTags = new List<ShaderTagId>() { k_LegacyPassName, k_CombinedRenderingPassName };

        private static readonly ProfilingSampler m_ProfilingDrawLights = new ProfilingSampler("Draw 2D Lights");
        private static readonly ProfilingSampler m_ProfilingDrawLightTextures = new ProfilingSampler("Draw 2D Lights Textures");
        private static readonly ProfilingSampler m_ProfilingDrawRenderers = new ProfilingSampler("Draw All Renderers");
        private static readonly ProfilingSampler m_ProfilingDrawLayerBatch = new ProfilingSampler("Draw Layer Batch");
        private static readonly ProfilingSampler m_ProfilingSamplerUnlit = new ProfilingSampler("Render Unlit");

        Material m_BlitMaterial;
        Material m_SamplingMaterial;

        private readonly Renderer2DData m_Renderer2DData;
        private bool m_NeedsDepth;
        private short m_CameraSortingLayerBoundsIndex;

        public Render2DLightingPass(Renderer2DData rendererData, Material blitMaterial, Material samplingMaterial)
        {
            m_Renderer2DData = rendererData;
            m_BlitMaterial = blitMaterial;
            m_SamplingMaterial = samplingMaterial;

            m_CameraSortingLayerBoundsIndex = GetCameraSortingLayerBoundsIndex();
        }

        internal void Setup(bool useDepth)
        {
            m_NeedsDepth = useDepth;
        }

        private void GetTransparencySortingMode(Camera camera, ref SortingSettings sortingSettings)
        {
            var mode = m_Renderer2DData.transparencySortMode;

            if (mode == TransparencySortMode.Default)
            {
                mode = camera.orthographic ? TransparencySortMode.Orthographic : TransparencySortMode.Perspective;
            }

            switch (mode)
            {
                case TransparencySortMode.Perspective:
                    sortingSettings.distanceMetric = DistanceMetric.Perspective;
                    break;
                case TransparencySortMode.Orthographic:
                    sortingSettings.distanceMetric = DistanceMetric.Orthographic;
                    break;
                default:
                    sortingSettings.distanceMetric = DistanceMetric.CustomAxis;
                    sortingSettings.customAxis = m_Renderer2DData.transparencySortAxis;
                    break;
            }
        }

        private void CopyCameraSortingLayerRenderTexture(ScriptableRenderContext context, RenderingData renderingData, RenderBufferStoreAction mainTargetStoreAction)
        {
            var cmd = CommandBufferPool.Get();
            cmd.Clear();
            this.CreateCameraSortingLayerRenderTexture(renderingData, cmd, m_Renderer2DData.cameraSortingLayerDownsamplingMethod);

            Material copyMaterial = m_Renderer2DData.cameraSortingLayerDownsamplingMethod == Downsampling._4xBox ? m_SamplingMaterial : m_BlitMaterial;
            RenderingUtils.Blit(cmd, colorAttachment, m_Renderer2DData.cameraSortingLayerRenderTarget.id, copyMaterial, 0, false, RenderBufferLoadAction.DontCare, RenderBufferStoreAction.Store, RenderBufferLoadAction.DontCare, RenderBufferStoreAction.DontCare);
            cmd.SetRenderTarget(colorAttachment, RenderBufferLoadAction.Load, mainTargetStoreAction,
                depthAttachment, RenderBufferLoadAction.Load, mainTargetStoreAction);
            cmd.SetGlobalTexture(k_CameraSortingLayerTextureID, m_Renderer2DData.cameraSortingLayerRenderTarget.id);
            context.ExecuteCommandBuffer(cmd);
            CommandBufferPool.Release(cmd);
        }

        private short GetCameraSortingLayerBoundsIndex()
        {
            SortingLayer[] sortingLayers = Light2DManager.GetCachedSortingLayer();
            for (short i = 0; i < sortingLayers.Length; i++)
            {
                if (sortingLayers[i].id == m_Renderer2DData.cameraSortingLayerTextureBound)
                    return (short)sortingLayers[i].value;
            }

            return short.MinValue;
        }

        private void DetermineWhenToResolve(int startIndex, int batchesDrawn, int batchCount, LayerBatch[] layerBatches,
            out int resolveDuringBatch, out bool resolveIsAfterCopy)
        {
            bool anyLightWithVolumetricShadows = false;
            var lights = m_Renderer2DData.lightCullResult.visibleLights;
            for (int i = 0; i < lights.Count; i++)
            {
                anyLightWithVolumetricShadows = lights[i].renderVolumetricShadows;
                if (anyLightWithVolumetricShadows)
                    break;
            }

            var lastVolumetricLightBatch = -1;
            if (anyLightWithVolumetricShadows)
            {
                for (int i = startIndex + batchesDrawn - 1; i >= startIndex; i--)
                {
                    if (layerBatches[i].lightStats.totalVolumetricUsage > 0)
                    {
                        lastVolumetricLightBatch = i;
                        break;
                    }
                }
            }

            if (m_Renderer2DData.useCameraSortingLayerTexture)
            {
                var cameraSortingLayerBoundsIndex = GetCameraSortingLayerBoundsIndex();
                var copyBatch = -1;
                for (int i = startIndex; i < startIndex + batchesDrawn; i++)
                {
                    var layerBatch = layerBatches[i];
                    if (cameraSortingLayerBoundsIndex >= layerBatch.layerRange.lowerBound && cameraSortingLayerBoundsIndex <= layerBatch.layerRange.upperBound)
                    {
                        copyBatch = i;
                        break;
                    }
                }

                resolveIsAfterCopy = copyBatch > lastVolumetricLightBatch;
                resolveDuringBatch = resolveIsAfterCopy ? copyBatch : lastVolumetricLightBatch;
            }
            else
            {
                resolveDuringBatch = lastVolumetricLightBatch;
                resolveIsAfterCopy = false;
            }
        }

        private void Render(ScriptableRenderContext context, CommandBuffer cmd, ref RenderingData renderingData, ref FilteringSettings filterSettings, DrawingSettings drawSettings)
        {
            var activeDebugHandler = GetActiveDebugHandler(renderingData);
            if (activeDebugHandler != null)
            {
                RenderStateBlock renderStateBlock = new RenderStateBlock();
                activeDebugHandler.DrawWithDebugRenderState(context, cmd, ref renderingData, ref drawSettings, ref filterSettings, ref renderStateBlock,
                    (ScriptableRenderContext ctx, ref RenderingData data, ref DrawingSettings ds, ref FilteringSettings fs, ref RenderStateBlock rsb) =>
                    {
                        ctx.DrawRenderers(data.cullResults, ref ds, ref fs, ref rsb);
                    });
            }
            else
            {
                context.DrawRenderers(renderingData.cullResults, ref drawSettings, ref filterSettings);
            }
        }

        private int DrawLayerBatches(
            LayerBatch[] layerBatches,
            int batchCount,
            int startIndex,
            CommandBuffer cmd,
            ScriptableRenderContext context,
            ref RenderingData renderingData,
            ref FilteringSettings filterSettings,
            ref DrawingSettings normalsDrawSettings,
            ref DrawingSettings drawSettings,
            ref RenderTextureDescriptor desc)
        {
            var debugHandler = GetActiveDebugHandler(renderingData);
            bool drawLights = debugHandler?.IsLightingActive ?? true;
            var batchesDrawn = 0;
            var rtCount = 0U;

            // Draw lights
            using (new ProfilingScope(cmd, m_ProfilingDrawLights))
            {
                for (var i = startIndex; i < batchCount; ++i)
                {
                    ref var layerBatch = ref layerBatches[i];

                    var blendStyleMask = layerBatch.lightStats.blendStylesUsed;
                    var blendStyleCount = 0U;
                    while (blendStyleMask > 0)
                    {
                        blendStyleCount += blendStyleMask & 1;
                        blendStyleMask >>= 1;
                    }

                    rtCount += blendStyleCount;

                    if (rtCount > LayerUtility.maxTextureCount)
                        break;

                    batchesDrawn++;

                    if (layerBatch.lightStats.totalNormalMapUsage > 0)
                    {
                        filterSettings.sortingLayerRange = layerBatch.layerRange;
                        var depthTarget = m_NeedsDepth ? depthAttachment : BuiltinRenderTextureType.None;
                        this.RenderNormals(context, renderingData, normalsDrawSettings, filterSettings, depthTarget, cmd, layerBatch.lightStats);
                    }

                    using (new ProfilingScope(cmd, m_ProfilingDrawLightTextures))
                    {
                        this.RenderLights(renderingData, cmd, layerBatch.startLayerID, ref layerBatch, ref desc);
                    }
                }
            }

            // Determine when to resolve in case we use MSAA
            var msaaEnabled = renderingData.cameraData.cameraTargetDescriptor.msaaSamples > 1;
            var isFinalBatchSet = startIndex + batchesDrawn >= batchCount;
            var resolveDuringBatch = -1;
            var resolveIsAfterCopy = false;
            if (msaaEnabled && isFinalBatchSet)
                DetermineWhenToResolve(startIndex, batchesDrawn, batchCount, layerBatches, out resolveDuringBatch, out resolveIsAfterCopy);


            // Draw renderers
            var blendStylesCount = m_Renderer2DData.lightBlendStyles.Length;
            using (new ProfilingScope(cmd, m_ProfilingDrawRenderers))
            {
                RenderBufferStoreAction initialStoreAction;
                if (msaaEnabled)
                    initialStoreAction = resolveDuringBatch < startIndex ? RenderBufferStoreAction.Resolve : RenderBufferStoreAction.StoreAndResolve;
                else
                    initialStoreAction = RenderBufferStoreAction.Store;
                cmd.SetRenderTarget(colorAttachment, RenderBufferLoadAction.Load, initialStoreAction, depthAttachment, RenderBufferLoadAction.Load, initialStoreAction);

                for (var i = startIndex; i < startIndex + batchesDrawn; i++)
                {
                    using (new ProfilingScope(cmd, m_ProfilingDrawLayerBatch))
                    {
                        // This is a local copy of the array element (it's a struct). Remember to add a ref here if you need to modify the real thing.
                        var layerBatch = layerBatches[i];

                        if (layerBatch.lightStats.totalLights > 0)
                        {
                            for (var blendStyleIndex = 0; blendStyleIndex < blendStylesCount; blendStyleIndex++)
                            {
                                var blendStyleMask = (uint)(1 << blendStyleIndex);
                                var blendStyleUsed = (layerBatch.lightStats.blendStylesUsed & blendStyleMask) > 0;

                                if (blendStyleUsed)
                                {
                                    var identifier = layerBatch.GetRTId(cmd, desc, blendStyleIndex);
                                    cmd.SetGlobalTexture(k_ShapeLightTextureIDs[blendStyleIndex], identifier);
                                }

                                RendererLighting.EnableBlendStyle(cmd, blendStyleIndex, blendStyleUsed);
                            }
                        }
                        else
                        {
                            for (var blendStyleIndex = 0; blendStyleIndex < k_ShapeLightTextureIDs.Length; blendStyleIndex++)
                            {
                                cmd.SetGlobalTexture(k_ShapeLightTextureIDs[blendStyleIndex], Texture2D.blackTexture);
                                RendererLighting.EnableBlendStyle(cmd, blendStyleIndex, blendStyleIndex == 0);
                            }
                        }

                        context.ExecuteCommandBuffer(cmd);
                        cmd.Clear();

                        short cameraSortingLayerBoundsIndex = GetCameraSortingLayerBoundsIndex();

                        RenderBufferStoreAction copyStoreAction;
                        if (msaaEnabled)
                            copyStoreAction = resolveDuringBatch == i && resolveIsAfterCopy ? RenderBufferStoreAction.Resolve : RenderBufferStoreAction.StoreAndResolve;
                        else
                            copyStoreAction = RenderBufferStoreAction.Store;
                        // If our camera sorting layer texture bound is inside our batch we need to break up the DrawRenderers into two batches
                        if (cameraSortingLayerBoundsIndex >= layerBatch.layerRange.lowerBound && cameraSortingLayerBoundsIndex < layerBatch.layerRange.upperBound && m_Renderer2DData.useCameraSortingLayerTexture)
                        {
                            filterSettings.sortingLayerRange = new SortingLayerRange(layerBatch.layerRange.lowerBound, cameraSortingLayerBoundsIndex);
                            Render(context, cmd, ref renderingData, ref filterSettings, drawSettings);
                            CopyCameraSortingLayerRenderTexture(context, renderingData, copyStoreAction);

                            filterSettings.sortingLayerRange = new SortingLayerRange((short)(cameraSortingLayerBoundsIndex + 1), layerBatch.layerRange.upperBound);
                            Render(context, cmd, ref renderingData, ref filterSettings, drawSettings);
                        }
                        else
                        {
                            filterSettings.sortingLayerRange = new SortingLayerRange(layerBatch.layerRange.lowerBound, layerBatch.layerRange.upperBound);
                            Render(context, cmd, ref renderingData, ref filterSettings, drawSettings);

                            if (cameraSortingLayerBoundsIndex == layerBatch.layerRange.upperBound && m_Renderer2DData.useCameraSortingLayerTexture)
                                CopyCameraSortingLayerRenderTexture(context, renderingData, copyStoreAction);
                        }

                        // Draw light volumes
                        if (drawLights && (layerBatch.lightStats.totalVolumetricUsage > 0))
                        {
                            var sampleName = "Render 2D Light Volumes";
                            cmd.BeginSample(sampleName);

                            RenderBufferStoreAction storeAction;
                            if (msaaEnabled)
                                storeAction = resolveDuringBatch == i && !resolveIsAfterCopy ? RenderBufferStoreAction.Resolve : RenderBufferStoreAction.StoreAndResolve;
                            else
                                storeAction = RenderBufferStoreAction.Store;
                            this.RenderLightVolumes(renderingData, cmd, layerBatch.startLayerID, layerBatch.endLayerValue, colorAttachment, depthAttachment,
                                RenderBufferStoreAction.Store, storeAction, false, m_Renderer2DData.lightCullResult.visibleLights);

                            cmd.EndSample(sampleName);
                        }
                    }
                }
            }

            for (var i = startIndex; i < startIndex + batchesDrawn; ++i)
            {
                ref var layerBatch = ref layerBatches[i];
                layerBatch.ReleaseRT(cmd);
            }

            return batchesDrawn;
        }

        private void UpdateCorners(Vector3 point, ref Vector3 minCorner, ref Vector3 maxCorner)
        {
            if (point.x < minCorner.x)
                minCorner.x = point.x;
            if (point.y < minCorner.y)
                minCorner.y = point.y;
            if (point.z < minCorner.z)
                minCorner.z = point.z;

            if (point.x > maxCorner.x)
                maxCorner.x = point.x;
            if (point.y > maxCorner.y)
                maxCorner.y = point.y;
            if (point.z > maxCorner.z)
                maxCorner.z = point.z;
        }

        private Matrix4x4 CalculateCameraLightFrustum(Camera camera, ILight2DCullResult cullResult)
        {
            const int k_Corners = 4;
            Vector2 planeSize = camera.GetFrustumPlaneSizeAt(camera.nearClipPlane);

            Vector3[] nearCorners = new Vector3[k_Corners];
            Vector3[] farCorners = new Vector3[k_Corners];
            camera.CalculateFrustumCorners(camera.rect, camera.nearClipPlane, camera.stereoActiveEye, nearCorners);
            camera.CalculateFrustumCorners(camera.rect, camera.farClipPlane, camera.stereoActiveEye, nearCorners);

            Vector3 minCorner = new Vector3(float.MaxValue, float.MaxValue, float.MaxValue);
            Vector3 maxCorner = new Vector3(float.MinValue, float.MinValue, float.MinValue);
            for (int i=0;i < k_Corners; i++)
            {
                UpdateCorners(nearCorners[i], ref minCorner, ref maxCorner);
                UpdateCorners(farCorners[i], ref minCorner, ref maxCorner);
            }

            List<Light2D> visibleLights = cullResult.visibleLights;
            for(int i=0;i<visibleLights.Count;i++)
                UpdateCorners(camera.transform.InverseTransformPoint(visibleLights[i].transform.position), ref minCorner, ref maxCorner);

            
            Matrix4x4 cameraLightFrustum = Matrix4x4.Ortho(minCorner.x, maxCorner.x, minCorner.y, maxCorner.y, minCorner.z, maxCorner.z);
            return cameraLightFrustum;
        }

        private void CallOnBeforeRender(Camera camera, ILight2DCullResult cullResult)
        {
            if (ShadowCasterGroup2DManager.shadowCasterGroups != null)
            {
                //Matrix4x4 cameraLightFrustum = CalculateCameraLightFrustum(camera, cullResult);
                Matrix4x4 cameraLightFrustum = camera.projectionMatrix;

                List<ShadowCasterGroup2D> groups = ShadowCasterGroup2DManager.shadowCasterGroups;
                for (int groupIndex = 0; groupIndex < groups.Count; groupIndex++)
                {
                    ShadowCasterGroup2D group = groups[groupIndex];

                    List<ShadowCaster2D> shadowCasters = group.GetShadowCasters();
                    for (int shadowCasterIndex = 0; shadowCasterIndex < shadowCasters.Count; shadowCasterIndex++)
                    {
                        ShadowCaster2D shadowCaster = shadowCasters[shadowCasterIndex];

                        if (shadowCaster.shadowCastingSource == ShadowCaster2D.ShadowCastingSources.ShapeProvider)
                        {
                            IShadowShape2DProvider provider = shadowCaster.shadowShape2DProvider;
                            provider.OnBeforeRender(shadowCaster.m_ShadowMesh, cameraLightFrustum);
                        }
                    }
                }
            }
        }

        public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
        {
            var isLitView = true;

#if UNITY_EDITOR
            if (renderingData.cameraData.isSceneViewCamera)
                isLitView = UnityEditor.SceneView.currentDrawingSceneView.sceneLighting;

            if (renderingData.cameraData.camera.cameraType == CameraType.Preview)
                isLitView = false;
#endif
            var camera = renderingData.cameraData.camera;
            var filterSettings = new FilteringSettings();
            filterSettings.renderQueueRange = RenderQueueRange.all;
            filterSettings.layerMask = -1;
            filterSettings.renderingLayerMask = 0xFFFFFFFF;
            filterSettings.sortingLayerRange = SortingLayerRange.all;

            LayerUtility.InitializeBudget(m_Renderer2DData.lightRenderTextureMemoryBudget);
            ShadowRendering.InitializeBudget(m_Renderer2DData.shadowRenderTextureMemoryBudget);

            var isSceneLit = m_Renderer2DData.lightCullResult.IsSceneLit();
            if (isSceneLit)
            {
                var combinedDrawSettings = CreateDrawingSettings(k_ShaderTags, ref renderingData, SortingCriteria.CommonTransparent);
                var normalsDrawSettings = CreateDrawingSettings(k_NormalsRenderingPassName, ref renderingData, SortingCriteria.CommonTransparent);

                var sortSettings = combinedDrawSettings.sortingSettings;
                GetTransparencySortingMode(camera, ref sortSettings);
                combinedDrawSettings.sortingSettings = sortSettings;
                normalsDrawSettings.sortingSettings = sortSettings;

                var cmd = CommandBufferPool.Get();
                cmd.SetGlobalFloat(k_HDREmulationScaleID, m_Renderer2DData.hdrEmulationScale);
                cmd.SetGlobalFloat(k_InverseHDREmulationScaleID, 1.0f / m_Renderer2DData.hdrEmulationScale);
                cmd.SetGlobalFloat(k_UseSceneLightingID, isLitView ? 1.0f : 0.0f);
                cmd.SetGlobalColor(k_RendererColorID, Color.white);
                this.SetShapeLightShaderGlobals(cmd);

                var desc = this.GetBlendStyleRenderTextureDesc(renderingData);

                CallOnBeforeRender(renderingData.cameraData.camera, m_Renderer2DData.lightCullResult);

                var layerBatches = LayerUtility.CalculateBatches(m_Renderer2DData.lightCullResult, out var batchCount);
                var batchesDrawn = 0;

                for (var i = 0; i < batchCount; i += batchesDrawn)
                    batchesDrawn = DrawLayerBatches(layerBatches, batchCount, i, cmd, context, ref renderingData, ref filterSettings, ref normalsDrawSettings, ref combinedDrawSettings, ref desc);

                this.DisableAllKeywords(cmd);
                this.ReleaseRenderTextures(cmd);
                context.ExecuteCommandBuffer(cmd);
                CommandBufferPool.Release(cmd);
            }
            else
            {
                var unlitDrawSettings = CreateDrawingSettings(k_ShaderTags, ref renderingData, SortingCriteria.CommonTransparent);
                var msaaEnabled = renderingData.cameraData.cameraTargetDescriptor.msaaSamples > 1;
                var storeAction = msaaEnabled ? RenderBufferStoreAction.Resolve : RenderBufferStoreAction.Store;

                var sortSettings = unlitDrawSettings.sortingSettings;
                GetTransparencySortingMode(camera, ref sortSettings);
                unlitDrawSettings.sortingSettings = sortSettings;

                var cmd = CommandBufferPool.Get();
                using (new ProfilingScope(cmd, m_ProfilingSamplerUnlit))
                {
                    cmd.SetRenderTarget(colorAttachment, RenderBufferLoadAction.Load, storeAction, depthAttachment, RenderBufferLoadAction.Load, storeAction);

                    cmd.SetGlobalFloat(k_UseSceneLightingID, isLitView ? 1.0f : 0.0f);
                    cmd.SetGlobalColor(k_RendererColorID, Color.white);

                    for (var blendStyleIndex = 0; blendStyleIndex < k_ShapeLightTextureIDs.Length; blendStyleIndex++)
                    {
                        if (blendStyleIndex == 0)
                            cmd.SetGlobalTexture(k_ShapeLightTextureIDs[blendStyleIndex], Texture2D.blackTexture);

                        RendererLighting.EnableBlendStyle(cmd, blendStyleIndex, blendStyleIndex == 0);
                    }
                }

                this.DisableAllKeywords(cmd);
                context.ExecuteCommandBuffer(cmd);

                Profiler.BeginSample("Render Sprites Unlit");
                if (m_Renderer2DData.useCameraSortingLayerTexture)
                {
                    filterSettings.sortingLayerRange = new SortingLayerRange(short.MinValue, m_CameraSortingLayerBoundsIndex);
                    Render(context, cmd, ref renderingData, ref filterSettings, unlitDrawSettings);

                    CopyCameraSortingLayerRenderTexture(context, renderingData, storeAction);

                    filterSettings.sortingLayerRange = new SortingLayerRange(m_CameraSortingLayerBoundsIndex, short.MaxValue);
                    Render(context, cmd, ref renderingData, ref filterSettings, unlitDrawSettings);
                }
                else
                {
                    Render(context, cmd, ref renderingData, ref filterSettings, unlitDrawSettings);
                }
                Profiler.EndSample();

                CommandBufferPool.Release(cmd);
            }

            filterSettings.sortingLayerRange = SortingLayerRange.all;
            RenderingUtils.RenderObjectsWithError(context, ref renderingData.cullResults, camera, filterSettings, SortingCriteria.None);
        }

        Renderer2DData IRenderPass2D.rendererData
        {
            get { return m_Renderer2DData; }
        }
    }
}
