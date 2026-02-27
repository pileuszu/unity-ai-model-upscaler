using UnityEngine;
using System.Collections.Generic;

namespace AiUpscaler.Runtime
{
    /// <summary>
    /// Handles Mesh Subdivision and Refinement.
    /// </summary>
    public static class MeshProcessor
    {
        /// <summary>
        /// Subdivides the given mesh by splitting each triangle into 4 smaller ones.
        /// </summary>
        public static Mesh Subdivide(Mesh originalMesh)
        {
            Vector3[] vertices = originalMesh.vertices;
            int[] triangles = originalMesh.triangles;
            Vector2[] uvs = originalMesh.uv;
            Vector3[] normals = originalMesh.normals;

            List<Vector3> newVertices = new List<Vector3>(vertices);
            List<Vector2> newUvs = new List<Vector2>(uvs);
            List<Vector3> newNormals = new List<Vector3>(normals);
            List<int> newTriangles = new List<int>();

            Dictionary<long, int> midPointCache = new Dictionary<long, int>();

            for (int i = 0; i < triangles.Length; i += 3)
            {
                int v1 = triangles[i];
                int v2 = triangles[i + 1];
                int v3 = triangles[i + 2];

                int a = GetMidPoint(v1, v2, ref newVertices, ref newUvs, ref newNormals, midPointCache);
                int b = GetMidPoint(v2, v3, ref newVertices, ref newUvs, ref newNormals, midPointCache);
                int c = GetMidPoint(v3, v1, ref newVertices, ref newUvs, ref newNormals, midPointCache);

                // Create 4 triangles from 1
                newTriangles.Add(v1); newTriangles.Add(a); newTriangles.Add(c);
                newTriangles.Add(v2); newTriangles.Add(b); newTriangles.Add(a);
                newTriangles.Add(v3); newTriangles.Add(c); newTriangles.Add(b);
                newTriangles.Add(a); newTriangles.Add(b); newTriangles.Add(c);
            }

            Mesh subdividedMesh = new Mesh();
            subdividedMesh.name = originalMesh.name + "_Subdivided";
            subdividedMesh.vertices = newVertices.ToArray();
            subdividedMesh.triangles = newTriangles.ToArray();
            subdividedMesh.uv = newUvs.ToArray();
            subdividedMesh.normals = newNormals.ToArray();
            
            subdividedMesh.RecalculateBounds();
            subdividedMesh.RecalculateTangents();

            return subdividedMesh;
        }

        private static int GetMidPoint(int v1, int v2, ref List<Vector3> vertices, ref List<Vector2> uvs, ref List<Vector3> normals, Dictionary<long, int> cache)
        {
            long key = ((long)Mathf.Min(v1, v2) << 32) | (long)Mathf.Max(v1, v2);
            if (cache.TryGetValue(key, out int index)) return index;

            Vector3 midPos = (vertices[v1] + vertices[v2]) * 0.5f;
            Vector2 midUv = (uvs[v1] + uvs[v2]) * 0.5f;
            Vector3 midNormal = (normals[v1] + normals[v2]).normalized;

            int newIndex = vertices.Count;
            vertices.Add(midPos);
            uvs.Add(midUv);
            normals.Add(midNormal);

            cache[key] = newIndex;
            return newIndex;
        }

        /// <summary>
        /// (Future Hook) Refines vertex positions using AI Inference.
        /// </summary>
        public static void ApplyNeuralRefinement(Mesh mesh, Unity.Sentis.ModelAsset onnxModel)
        {
            // TODO: Use Sentis to predict vertex offsets for better curvature
            Debug.Log("[AI Upscaler] Neural Mesh Refinement not yet implemented.");
        }
    }
}
