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
            BoneWeight[] boneWeights = originalMesh.boneWeights;

            List<Vector3> newVertices = new List<Vector3>(vertices);
            List<Vector2> newUvs = new List<Vector2>(uvs);
            List<Vector3> newNormals = new List<Vector3>(normals);
            List<BoneWeight> newBoneWeights = (boneWeights != null && boneWeights.Length > 0) ? new List<BoneWeight>(boneWeights) : null;
            List<int> newTriangles = new List<int>();

            Dictionary<long, int> midPointCache = new Dictionary<long, int>();

            for (int i = 0; i < triangles.Length; i += 3)
            {
                int v1 = triangles[i];
                int v2 = triangles[i + 1];
                int v3 = triangles[i + 2];

                int a = GetMidPoint(v1, v2, ref newVertices, ref newUvs, ref newNormals, ref newBoneWeights, midPointCache);
                int b = GetMidPoint(v2, v3, ref newVertices, ref newUvs, ref newNormals, ref newBoneWeights, midPointCache);
                int c = GetMidPoint(v3, v1, ref newVertices, ref newUvs, ref newNormals, ref newBoneWeights, midPointCache);

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
            if (newBoneWeights != null) subdividedMesh.boneWeights = newBoneWeights.ToArray();
            subdividedMesh.bindposes = originalMesh.bindposes;

            TransferBlendShapes(originalMesh, subdividedMesh, midPointCache);
            
            subdividedMesh.RecalculateBounds();
            subdividedMesh.RecalculateNormals(); // Crucial for fixing dark shading/black face issues
            subdividedMesh.RecalculateTangents();

            return subdividedMesh;
        }

        private static int GetMidPoint(int v1, int v2, ref List<Vector3> vertices, ref List<Vector2> uvs, ref List<Vector3> normals, ref List<BoneWeight> boneWeights, Dictionary<long, int> cache)
        {
            long key = ((long)Mathf.Min(v1, v2) << 32) | (long)Mathf.Max(v1, v2);
            if (cache.TryGetValue(key, out int index)) return index;

            Vector3 midPos = (vertices[v1] + vertices[v2]) * 0.5f;
            Vector2 midUv = (uvs[v1] + uvs[v2]) * 0.5f;
            Vector3 midNormal = (normals[v1] + normals[v2]).normalized;

            if (boneWeights != null)
            {
                boneWeights.Add(InterpolateBoneWeight(boneWeights[v1], boneWeights[v2]));
            }

            int newIndex = vertices.Count;
            vertices.Add(midPos);
            uvs.Add(midUv);
            normals.Add(midNormal);

            cache[key] = newIndex;
            return newIndex;
        }

        private static BoneWeight InterpolateBoneWeight(BoneWeight bw1, BoneWeight bw2)
        {
            BoneWeight result = new BoneWeight();
            result.boneIndex0 = bw1.boneIndex0; result.weight0 = bw1.weight0 * 0.5f;
            result.boneIndex1 = bw1.boneIndex1; result.weight1 = bw1.weight1 * 0.5f;
            result.boneIndex2 = bw1.boneIndex2; result.weight2 = bw1.weight2 * 0.5f;
            result.boneIndex3 = bw1.boneIndex3; result.weight3 = bw1.weight3 * 0.5f;

            result.weight0 += bw2.weight0 * 0.5f;
            result.weight1 += bw2.weight1 * 0.5f;
            result.weight2 += bw2.weight2 * 0.5f;
            result.weight3 += bw2.weight3 * 0.5f;
            return result;
        }

        private static void TransferBlendShapes(Mesh source, Mesh target, Dictionary<long, int> midPointCache)
        {
            int shapeCount = source.blendShapeCount;
            for (int i = 0; i < shapeCount; i++)
            {
                string shapeName = source.GetBlendShapeName(i);
                int frameCount = source.GetBlendShapeFrameCount(i);
                for (int f = 0; f < frameCount; f++)
                {
                    float frameWeight = source.GetBlendShapeFrameWeight(i, f);
                    Vector3[] dv = new Vector3[source.vertexCount];
                    Vector3[] dn = new Vector3[source.vertexCount];
                    Vector3[] dt = new Vector3[source.vertexCount];
                    source.GetBlendShapeFrameVertices(i, f, dv, dn, dt);

                    Vector3[] newDv = new Vector3[target.vertexCount];
                    Vector3[] newDn = new Vector3[target.vertexCount];
                    Vector3[] newDt = new Vector3[target.vertexCount];

                    System.Array.Copy(dv, newDv, dv.Length);
                    System.Array.Copy(dn, newDn, dn.Length);
                    System.Array.Copy(dt, newDt, dt.Length);

                    foreach (var pair in midPointCache)
                    {
                        int v1_idx = (int)(pair.Key >> 32);
                        int v2_idx = (int)(pair.Key & 0xFFFFFFFF);
                        int newV = pair.Value;

                        newDv[newV] = (dv[v1_idx] + dv[v2_idx]) * 0.5f;
                        newDn[newV] = (dn[v1_idx] + dn[v2_idx]) * 0.5f;
                        newDt[newV] = (dt[v1_idx] + dt[v2_idx]) * 0.5f;
                    }
                    target.AddBlendShapeFrame(shapeName, frameWeight, newDv, newDn, newDt);
                }
            }
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
