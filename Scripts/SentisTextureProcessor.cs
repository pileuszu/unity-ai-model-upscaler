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
            int tileSize = input.shape[3].value > 0 ? input.shape[3].value : 512;
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

            // Tiling Settings with Overlap
            int padding = 12; // Context padding on each side
            int stepSize = tileSize - (padding * 2);
            if (stepSize <= 0) stepSize = tileSize / 2;

            // Temp Texture to extract patches
            Texture2D patchTex = new Texture2D(tileSize, tileSize, TextureFormat.RGBA32, false);

            for (int y = -padding; y < inHeight; y += stepSize)
            {
                for (int x = -padding; x < inWidth; x += stepSize)
                {
                    // 1. Extract Patch with Padding
                    // Clamp to source image boundaries
                    int extractX = Mathf.Clamp(x, 0, inWidth - tileSize);
                    if (inWidth < tileSize) extractX = 0;
                    
                    int extractY = Mathf.Clamp(y, 0, inHeight - tileSize);
                    if (inHeight < tileSize) extractY = 0;

                    int currentW = Mathf.Min(tileSize, inWidth);
                    int currentH = Mathf.Min(tileSize, inHeight);

                    Color[] pixels = inputTexture.GetPixels(extractX, extractY, currentW, currentH);
                    patchTex.Reinitialize(currentW, currentH);
                    patchTex.SetPixels(pixels);
                    patchTex.Apply();

                    // 2. To Tensor
                    using TensorFloat inputTensor = TextureConverter.ToTensor(patchTex, currentW, currentH, 3);

                    // 3. Inference
                    _engine.Execute(inputTensor);
                    TensorFloat outputTensor = _engine.PeekOutput() as TensorFloat;

                    // 4. Handle Output Tensor Rank (ShallowReshape for 1.2.x compatibility)
                    int patchOutW, patchOutH;
                    TensorFloat finalOutputTensor = outputTensor;
                    if (outputTensor.shape.rank == 4)
                    {
                        patchOutW = outputTensor.shape[3];
                        patchOutH = outputTensor.shape[2];
                    }
                    else if (outputTensor.shape.rank == 3)
                    {
                        patchOutW = outputTensor.shape[2];
                        patchOutH = outputTensor.shape[1];
                        var newShape = new TensorShape(1, outputTensor.shape[0], outputTensor.shape[1], outputTensor.shape[2]);
                        finalOutputTensor = outputTensor.ShallowReshape(newShape) as TensorFloat;
                    }
                    else continue;

                    RenderTexture patchRT = new RenderTexture(patchOutW, patchOutH, 0, RenderTextureFormat.ARGB32);
                    patchRT.enableRandomWrite = true;
                    patchRT.Create();
                    TextureConverter.RenderToTexture(finalOutputTensor, patchRT);

                    // 5. Calculate "Safe" Commit Region (Crop the padding)
                    float outScale = (float)patchOutW / currentW;
                    
                    // Region in the upscaled patch to copy
                    int offsetX = (x < 0) ? 0 : Mathf.RoundToInt(padding * outScale);
                    int offsetY = (y < 0) ? 0 : Mathf.RoundToInt(padding * outScale);
                    
                    int copyW = Mathf.RoundToInt(stepSize * outScale);
                    int copyH = Mathf.RoundToInt(stepSize * outScale);

                    // Destination in the big texture
                    int destX = Mathf.RoundToInt(Mathf.Max(0, x + padding) * outScale);
                    int destY = Mathf.RoundToInt(Mathf.Max(0, y + padding) * outScale);

                    // Cleanup for final edges
                    if (destX + copyW > outWidth) copyW = outWidth - destX;
                    if (destY + copyH > outHeight) copyH = outHeight - destY;
                    if (offsetX + copyW > patchOutW) copyW = patchOutW - offsetX;
                    if (offsetY + copyH > patchOutH) copyH = patchOutH - offsetY;

                    if (copyW > 0 && copyH > 0)
                    {
                        Graphics.CopyTexture(patchRT, 0, 0, offsetX, offsetY, copyW, copyH, finalRT, 0, 0, destX, destY);
                    }
                    
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
