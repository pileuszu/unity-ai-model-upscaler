using UnityEngine;
using Unity.Sentis;
using System.IO;

namespace AiUpscaler.Runtime
{
    /// <summary>
    /// Handles AI Inference for Texture Super-Resolution using Unity Sentis.
    /// </summary>
    public class SentisTextureProcessor
    {
        private Model _runtimeModel;
        private IWorker _engine;

        public SentisTextureProcessor(ModelAsset modelAsset)
        {
            // Load the ONNX model
            _runtimeModel = ModelLoader.Load(modelAsset);
            // Create a worker (engine) to run on GPU if available
            _engine = WorkerFactory.CreateWorker(BackendType.GPUCompute, _runtimeModel);
        }

        public Texture2D UpscaleTexture(Texture2D inputTexture, float scale)
        {
            if (inputTexture == null) return null;

            // Detect Model Input Shape (e.g., 128x128)
            var input = _runtimeModel.inputs[0];
            int tileSize = input.shape[3].IsMax ? 512 : input.shape[3].value;
            if (tileSize <= 0) tileSize = 512;

            int inWidth = inputTexture.width;
            int inHeight = inputTexture.height;
            int outWidth = Mathf.RoundToInt(inWidth * scale);
            int outHeight = Mathf.RoundToInt(inHeight * scale);

            Debug.Log($"[AI Upscaler] Tiling Mode Start: {inWidth}x{inHeight} -> {outWidth}x{outHeight} using {tileSize}x{tileSize} patches.");

            // Create final result texture
            RenderTexture finalRT = new RenderTexture(outWidth, outHeight, 0, RenderTextureFormat.ARGB32);
            finalRT.enableRandomWrite = true;
            finalRT.Create();

            // Temp Texture to extract patches
            Texture2D patchTex = new Texture2D(tileSize, tileSize, TextureFormat.RGBA32, false);

            for (int y = 0; y < inHeight; y += tileSize)
            {
                for (int x = 0; x < inWidth; x += tileSize)
                {
                    int currentW = Mathf.Min(tileSize, inWidth - x);
                    int currentH = Mathf.Min(tileSize, inHeight - y);

                    // 1. Extract Patch
                    Color[] pixels = inputTexture.GetPixels(x, y, currentW, currentH);
                    patchTex.Reinitialize(currentW, currentH);
                    patchTex.SetPixels(pixels);
                    patchTex.Apply();

                    // 2. To Tensor
                    using TensorFloat inputTensor = TextureConverter.ToTensor(patchTex, currentW, currentH, 3);

                    // 3. Inference
                    _engine.Execute(inputTensor);
                    TensorFloat outputTensor = _engine.PeekOutput() as TensorFloat;

                    // 4. To Temp RT
                    int patchOutW = outputTensor.shape[3];
                    int patchOutH = outputTensor.shape[2];
                    RenderTexture patchRT = new RenderTexture(patchOutW, patchOutH, 0, RenderTextureFormat.ARGB32);
                    patchRT.enableRandomWrite = true;
                    patchRT.Create();

                    TextureConverter.RenderToTexture(outputTensor, patchRT);

                    // 5. Blit to Final RT
                    Graphics.CopyTexture(patchRT, 0, 0, 0, 0, patchOutW, patchOutH, finalRT, 0, 0, Mathf.RoundToInt(x * scale), Mathf.RoundToInt(y * scale));
                    
                    patchRT.Release();
                    Object.DestroyImmediate(patchRT);
                }
            }

            // Copy Final RT to Texture2D
            Texture2D resultTexture = new Texture2D(outWidth, outHeight, TextureFormat.RGBA32, false);
            RenderTexture.active = finalRT;
            resultTexture.ReadPixels(new Rect(0, 0, outWidth, outHeight), 0, 0);
            resultTexture.Apply();
            RenderTexture.active = null;

            finalRT.Release();
            Object.DestroyImmediate(finalRT);
            Object.DestroyImmediate(patchTex);

            return resultTexture;
        }

        public void Release()
        {
            _engine?.Dispose();
        }
    }
}
