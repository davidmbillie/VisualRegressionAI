using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SixLabors.Fonts;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SixLabors.ImageSharp.Drawing.Processing;
using System.Numerics;

public class ObjectDetectionInput
{
    public string ImagePath { get; set; }
}

public class ObjectDetectionOutput
{
    [ColumnName("boxes")]
    public float[] Boxes { get; set; }

    [ColumnName("labels")]
    public float[] Labels { get; set; }

    [ColumnName("scores")]
    public long[] Scores { get; set; }
}

public class ObjectDetector
{
    private readonly MLContext _mlContext;
    private readonly ITransformer _model;

    public ObjectDetector(string modelPath)
    {
        _mlContext = new MLContext();

        // Dummy data to initialize schema
        var sample = new List<ObjectDetectionInput> { new ObjectDetectionInput { ImagePath = "placeholder.jpg" } };
        var dummyData = _mlContext.Data.LoadFromEnumerable(sample);

        var pipeline = _mlContext.Transforms.LoadImages("input", "", nameof(ObjectDetectionInput.ImagePath))
            .Append(_mlContext.Transforms.ResizeImages("input", imageWidth: 320, imageHeight: 320))
            .Append(_mlContext.Transforms.ExtractPixels("input"))
            .Append(_mlContext.Transforms.ApplyOnnxModel(
                modelFile: modelPath,
                outputColumnNames: new[] { "boxes", "labels", "scores" },
                inputColumnNames: new[] { "input" }));

        _model = pipeline.Fit(dummyData);
    }

    public void Predict(string imagePath)
    {
        var data = new List<ObjectDetectionInput> { new ObjectDetectionInput { ImagePath = imagePath } };
        var imageData = _mlContext.Data.LoadFromEnumerable(data);

        var prediction = _model.Transform(imageData);
        var results = _mlContext.Data.CreateEnumerable<ObjectDetectionOutput>(prediction, reuseRowObject: false).ToList();

        using Image<Rgba32> image = Image.Load<Rgba32>(imagePath);
        Font font = SystemFonts.CreateFont("Arial", 12);

        foreach (var result in results)
        {
            for (int i = 0; i < result.Scores.Length; i++)
            {
                if (result.Scores[i] > 0.5f)
                {
                    int idx = i * 4;
                    float xMin = result.Boxes[idx];
                    float yMin = result.Boxes[idx + 1];
                    float xMax = result.Boxes[idx + 2];
                    float yMax = result.Boxes[idx + 3];

                    float width = xMax - xMin;
                    float height = yMax - yMin;

                    var rect = new RectangleF(xMin, yMin, width, height);
                    var label = $"{result.Labels[i]} ({result.Scores[i]:F2})";
                    var labelPos = new PointF(xMin, Math.Max(yMin - 20, 0));

                    image.Mutate(ctx =>
                    {
                        ctx.Draw(Color.Red, 2f, rect);
                        ctx.DrawText(label, font, Color.Red, labelPos);
                    });
                }
            }
        }

        var outputPath = Path.Combine(Path.GetDirectoryName(imagePath)!, "annotated_output.png");
        image.Save(outputPath);
        Console.WriteLine($"🎯 Saved to: {outputPath}");
    }
}
