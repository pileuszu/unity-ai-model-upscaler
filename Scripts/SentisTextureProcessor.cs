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

            // 1. Convert Texture to Tensor
            // Sentis expects values usually normalized or in specific shapes.
            // For ESRGAN, it's often [1, 3, H, W]
            using TensorFloat inputTensor = TextureConverter.ToTensor(inputTexture, inputTexture.width, inputTexture.height, 3);

            // 2. Run Inference
            _engine.Execute(inputTensor);

            // 3. Get Result Tensor
            TensorFloat outputTensor = _engine.PeekOutput() as TensorFloat;

            // 4. Convert Tensor back to Texture
            // We need to calculate the output dimensions based on the scale
            int outWidth = Mathf.RoundToInt(inputTexture.width * scale);
            int outHeight = Mathf.RoundToInt(inputTexture.height * scale);

            Texture2D resultTexture = new Texture2D(outWidth, outHeight, TextureFormat.RGBA32, false);
            TextureConverter.RenderToTexture(outputTensor, resultTexture);

            return resultTexture;
        }

        public void Release()
        {
            _engine?.Dispose();
        }
    }
}
