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

            // 1. Detect Model Input Shape
            var input = _runtimeModel.inputs[0];
            int modelWidth = input.shape[3].IsMax ? inputTexture.width : input.shape[3].value;
            int modelHeight = input.shape[2].IsMax ? inputTexture.height : input.shape[2].value;

            // Handle cases where model shape is fixed (e.g., 128x128)
            if (modelWidth <= 0) modelWidth = 512; // Fallback
            if (modelHeight <= 0) modelHeight = 512;

            Debug.Log($"[AI Upscaler] Model Input Shape: {modelWidth}x{modelHeight}. Upscaling from {inputTexture.width}x{inputTexture.height}");

            // 2. Convert Texture to Tensor (Auto-resizes if dimensions differ)
            using TensorFloat inputTensor = TextureConverter.ToTensor(inputTexture, modelWidth, modelHeight, 3);

            // 3. Run Inference
            try 
            {
                _engine.Execute(inputTensor);
            }
            catch (System.Exception e)
            {
                Debug.LogError($"[AI Upscaler] Inference Failed: {e.Message}\nTry a smaller texture or a lighter model.");
                return null;
            }

            // 4. Get Result Tensor
            TensorFloat outputTensor = _engine.PeekOutput() as TensorFloat;
            if (outputTensor == null) return null;

            // Calculate output dimensions based on output tensor shape
            int outWidth = outputTensor.shape[3];
            int outHeight = outputTensor.shape[2];

            Debug.Log($"[AI Upscaler] AI Output Shape: {outWidth}x{outHeight}");

            // 5. Convert Tensor back to Texture
            RenderTexture rt = new RenderTexture(outWidth, outHeight, 0, RenderTextureFormat.ARGB32);
            rt.enableRandomWrite = true;
            rt.Create();

            try 
            {
                TextureConverter.RenderToTexture(outputTensor, rt);
            }
            catch (System.Exception e)
            {
                Debug.LogError($"[AI Upscaler] RenderToTexture Failed: {e.Message}");
                rt.Release();
                return null;
            }

            // 6. Copy RenderTexture back to Texture2D
            Texture2D resultTexture = new Texture2D(outWidth, outHeight, TextureFormat.RGBA32, false);
            RenderTexture.active = rt;
            resultTexture.ReadPixels(new Rect(0, 0, outWidth, outHeight), 0, 0);
            resultTexture.Apply();
            RenderTexture.active = null;

            // Cleanup
            rt.Release();
            if (Application.isEditor) Object.DestroyImmediate(rt);
            else Object.Destroy(rt);

            return resultTexture;
        }

        public void Release()
        {
            _engine?.Dispose();
        }
    }
}
