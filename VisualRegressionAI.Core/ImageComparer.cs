using System;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;

public class ImageComparer
{
    private readonly MLContext mlContext;
    private readonly string modelPath;

    public ImageComparer(string modelPath)
    {
        mlContext = new MLContext();
        this.modelPath = modelPath;
    }

    public float CompareImages(string imagePath1, string imagePath2)
    {
        var data1 = GetImageEmbedding(imagePath1);
        var data2 = GetImageEmbedding(imagePath2);

        float dot = 0f, normA = 0f, normB = 0f;
        for (int i = 0; i < data1.Length; i++)
        {
            dot += data1[i] * data2[i];
            normA += data1[i] * data1[i];
            normB += data2[i] * data2[i];
        }
        return dot / (float)(Math.Sqrt(normA) * Math.Sqrt(normB));
    }

    private float[] GetImageEmbedding(string path)
    {
        var imageData = new[] { new InputData { ImagePath = path } };
        var data = mlContext.Data.LoadFromEnumerable(imageData);

        var pipeline = mlContext.Transforms.LoadImages("data", "", nameof(InputData.ImagePath))
            .Append(mlContext.Transforms.ResizeImages("data", 224, 224))
            .Append(mlContext.Transforms.ExtractPixels("data"))
            .Append(mlContext.Transforms.ApplyOnnxModel(
                modelFile: modelPath,
                outputColumnNames: new[] { "resnetv24_dense0_fwd" },
                inputColumnNames: new[] { "data" }
            ));

        var transformed = pipeline.Fit(data).Transform(data);
        var prediction = mlContext.Data.CreateEnumerable<OutputData>(transformed, reuseRowObject: false).First();
        return prediction.resnetv24_dense0_fwd;
    }

    private class InputData
    {
        public string ImagePath { get; set; }
    }

    private class OutputData
    {
        [VectorType(1000)]
        public float[] resnetv24_dense0_fwd { get; set; }
    }
}