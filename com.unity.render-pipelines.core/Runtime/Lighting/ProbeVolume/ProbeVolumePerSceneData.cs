using System;
using System.Collections.Generic;

#if UNITY_EDITOR
using System.IO;
using UnityEditor;
#endif

namespace UnityEngine.Experimental.Rendering
{
    /// <summary>
    /// A component that stores baked probe volume state and data references. Normally hidden from the user.
    /// </summary>
    [ExecuteAlways]
    [AddComponentMenu("")] // Hide.
    public class ProbeVolumePerSceneData : MonoBehaviour, ISerializationCallbackReceiver
    {
        [Serializable]
        internal struct PerStateData
        {
            public int sceneHash;
            public TextAsset cellDataAsset; // Contains L0 L1 SH data
            public TextAsset cellOptionalDataAsset; // Contains L2 SH data
        }

        [Serializable]
        struct SerializablePerStateDataItem
        {
            public string state;
            public PerStateData data;
        }

        [SerializeField] internal ProbeVolumeAsset asset;
        [SerializeField] internal TextAsset cellSharedDataAsset; // Contains bricks and validity data
        [SerializeField] internal TextAsset cellSupportDataAsset; // Contains debug data
        [SerializeField] List<SerializablePerStateDataItem> serializedStates = new();

        internal Dictionary<string, PerStateData> states = new();

        string currentState0 = null, currentState1 = null;

        /// <summary>
        /// OnAfterDeserialize implementation.
        /// </summary>
        void ISerializationCallbackReceiver.OnAfterDeserialize()
        {
            states.Clear();
            foreach (var stateData in serializedStates)
                states.Add(stateData.state, stateData.data);
        }

        /// <summary>
        /// OnBeforeSerialize implementation.
        /// </summary>
        void ISerializationCallbackReceiver.OnBeforeSerialize()
        {
            serializedStates.Clear();
            foreach (var kvp in states)
            {
                serializedStates.Add(new SerializablePerStateDataItem()
                {
                    state = kvp.Key,
                    data = kvp.Value,
                });
            }
        }

#if UNITY_EDITOR
        void DeleteAsset(Object asset)
        {
            if (AssetDatabase.TryGetGUIDAndLocalFileIdentifier(asset, out string guid, out long instanceID))
            {
                var assetPath = AssetDatabase.GUIDToAssetPath(guid);
                AssetDatabase.DeleteAsset(assetPath);
            }
        }
#endif

        internal void Clear()
        {
            QueueAssetRemoval();

#if UNITY_EDITOR
            try
            {
                AssetDatabase.StartAssetEditing();
                DeleteAsset(asset);
                DeleteAsset(cellSharedDataAsset);
                DeleteAsset(cellSupportDataAsset);
                foreach (var stateData in states.Values)
                {
                    DeleteAsset(stateData.cellDataAsset);
                    DeleteAsset(stateData.cellOptionalDataAsset);
                }
            }
            finally
            {
                AssetDatabase.StopAssetEditing();
                AssetDatabase.Refresh();
                EditorUtility.SetDirty(this);
            }
#endif

            states.Clear();
        }

        internal void RemoveBakingState(string state)
        {
#if UNITY_EDITOR
            if (states.TryGetValue(state, out var stateData))
            {
                AssetDatabase.DeleteAsset(AssetDatabase.GetAssetPath(stateData.cellDataAsset));
                AssetDatabase.DeleteAsset(AssetDatabase.GetAssetPath(stateData.cellOptionalDataAsset));
                EditorUtility.SetDirty(this);
            }
#endif
            states.Remove(state);
        }

        internal void RenameBakingState(string state, string newState)
        {
            if (!states.TryGetValue(state, out var stateData))
                return;
            states.Remove(state);
            states.Add(newState, stateData);

#if UNITY_EDITOR
            EditorUtility.SetDirty(this);
            var baseName = ProbeVolumeAsset.assetName + "-" + newState;
            void RenameAsset(Object asset, string extension)
            {
                var oldPath = AssetDatabase.GetAssetPath(asset);
                AssetDatabase.RenameAsset(oldPath, baseName + extension);
            }
            RenameAsset(stateData.cellDataAsset, ".CellData.bytes");
            RenameAsset(stateData.cellOptionalDataAsset, ".CellOptionalData.bytes");
#endif
        }

        internal bool ResolveCells() => ResolveSharedCellData() && ResolvePerStateCellData();

        bool ResolveSharedCellData() => asset != null && asset.ResolveSharedCellData(cellSharedDataAsset, cellSupportDataAsset);
        bool ResolvePerStateCellData()
        {
            if (currentState0 == null || !states.TryGetValue(currentState0, out var data0))
                return false;
            bool result = asset.ResolvePerStateCellData(0, data0.cellDataAsset, data0.cellOptionalDataAsset);
            if (currentState1 != null && states.TryGetValue(currentState1, out var data1))
                result = asset.ResolvePerStateCellData(1, data1.cellDataAsset, data1.cellOptionalDataAsset);
            return result;
        }

        internal void QueueAssetLoading()
        {
            if (asset == null || !ResolvePerStateCellData())
                return;

            var refVol = ProbeReferenceVolume.instance;
            refVol.AddPendingAssetLoading(asset);
#if UNITY_EDITOR
            if (refVol.sceneData != null)
                refVol.bakingProcessSettings = refVol.sceneData.GetBakeSettingsForScene(gameObject.scene);
#endif
        }

        internal void QueueAssetRemoval()
        {
            if (asset != null)
                ProbeReferenceVolume.instance.AddPendingAssetRemoval(asset);
        }

        void OnEnable()
        {
            ProbeReferenceVolume.instance.RegisterPerSceneData(this);

            ResolveSharedCellData();
            var prv = ProbeReferenceVolume.instance;
            if (prv.sceneData != null)
                SetBakingState(prv.bakingState0, prv.bakingState1);
            // otherwise baking state will be initialized in ProbeReferenceVolume.Initialize when sceneData is loaded
        }

        void OnDisable()
        {
            OnDestroy();
            ProbeReferenceVolume.instance.UnregisterPerSceneData(this);
        }

        void OnDestroy()
        {
            QueueAssetRemoval();
            currentState0 = currentState1 = null;
        }

        internal void SetBakingState(string state0, string state1)
        {
            if (state0 == currentState0 && state1 == currentState1)
                return;

            QueueAssetRemoval();
            currentState0 = state0;
            currentState1 = state1;
            QueueAssetLoading();
        }

#if UNITY_EDITOR
        internal void GetBlobFileNames(out string cellDataFilename, out string cellOptionalDataFilename, out string cellSharedDataFilename, out string cellSupportDataFilename)
        {
            var state = ProbeReferenceVolume.instance.bakingState;
            string basePath = Path.Combine(ProbeVolumeAsset.GetDirectory(gameObject.scene.path, gameObject.scene.name), ProbeVolumeAsset.assetName);

            string GetOrCreateFileName(Object o, string extension)
            {
                var res = AssetDatabase.GetAssetPath(o);
                if (string.IsNullOrEmpty(res)) res = basePath + extension;
                return res;
            }
            cellDataFilename = GetOrCreateFileName(states[state].cellDataAsset, "-" + state + ".CellData.bytes");
            cellOptionalDataFilename = GetOrCreateFileName(states[state].cellOptionalDataAsset, "-" + state + ".CellOptionalData.bytes");
            cellSharedDataFilename = GetOrCreateFileName(cellSharedDataAsset, ".CellSharedData.bytes");
            cellSupportDataFilename = GetOrCreateFileName(cellSupportDataAsset, ".CellSupportData.bytes");
        }

        /// <summary>
        /// Call this function during OnProcessScene to strip debug from project builds.
        /// </summary>
        public void StripSupportData()
        {
            cellSupportDataAsset = null;
        }
#endif
    }
}
