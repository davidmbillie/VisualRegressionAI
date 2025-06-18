using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using SixLabors.Fonts;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.ImageSharp.Processing; // For image.Mutate extensions
using SixLabors.ImageSharp.PixelFormats;  // For pixel formats

public class ObjectDetectionInput
{
    // Instead of holding a Bitmap, we hold the image file path.
    [ColumnName("input")]
    public string ImagePath { get; set; }
}

public class ObjectDetectionOutput
{
    [ColumnName("boxes")]
    public float[] Boxes { get; set; }

    [ColumnName("labels")]
    public long[] Labels { get; set; }

    [ColumnName("scores")]
    public float[] Scores { get; set; }
}

public class ObjectDetector
{
    private readonly string _modelPath;
    private readonly MLContext _mlContext;
    private readonly ITransformer _model;

    public ObjectDetector(string modelPath)
    {
        _modelPath = modelPath;
        _mlContext = new MLContext();
        
        var pipeline = _mlContext.Transforms.LoadImages(
                            outputColumnName: "input",
                            imageFolder: "",
                            inputColumnName: nameof(ObjectDetectionInput.ImagePath))
            .Append(_mlContext.Transforms.ResizeImages("input", 320, 320,
                            inputColumnName: "input",
                            resizing: Microsoft.ML.Transforms.Image.ImageResizingEstimator.ResizingKind.Fill))
            .Append(_mlContext.Transforms.ExtractPixels("input", interleavePixelColors: true))
            .Append(_mlContext.Transforms.ApplyOnnxModel(
                            modelFile: _modelPath,
                            outputColumnNames: new[] { "boxes", "labels", "scores" },
                            inputColumnNames: new[] { "input" }));

        // Fit the pipeline on an empty dataset.
        _model = pipeline.Fit(_mlContext.Data.LoadFromEnumerable(new List<ObjectDetectionInput>()));
    }

    public void Predict(string imagePath)
    {
        // Prepare ML.NET prediction input using the image path.
        var input = new ObjectDetectionInput { ImagePath = imagePath };
        var data = _mlContext.Data.LoadFromEnumerable(new[] { input });
        var predictions = _model.Transform(data);
        var results = _mlContext.Data.CreateEnumerable<ObjectDetectionOutput>(predictions, reuseRowObject: false);

        // Load the image using ImageSharp.
        using Image<Rgba32> image = Image.Load<Rgba32>(imagePath);

        // Prepare a font (using SixLabors.Fonts). Adjust the font family and size as needed.
        Font font = SystemFonts.CreateFont("Arial", 10);

        // Process each prediction result.
        foreach (var prediction in results)
        {
            // For each detected object, if confidence is above 0.5, draw a box and label.
            for (int i = 0; i < prediction.Scores.Length; i++)
            {
                if (prediction.Scores[i] > 0.5f)
                {
                    int idx = i * 4;
                    float xMin = prediction.Boxes[idx];
                    float yMin = prediction.Boxes[idx + 1];
                    float xMax = prediction.Boxes[idx + 2];
                    float yMax = prediction.Boxes[idx + 3];

                    float width = xMax - xMin;
                    float height = yMax - yMin;

                    // Define a rectangle for the bounding box.
                    var rect = new SixLabors.ImageSharp.RectangleF(xMin, yMin, width, height);

                    // Draw the bounding box with a red pen of thickness 2.
                    image.Mutate(ctx => ctx.Draw(
                        color: Color.Red,
                        thickness: 2,
                        shape: rect));

                    // Draw the label and score just above the box.
                    string labelText = $"{prediction.Labels[i]} ({prediction.Scores[i]:F2})";
                    // Ensure the text doesn't get drawn off the top of the image.
                    var textLocation = new SixLabors.ImageSharp.PointF(xMin, Math.Max(yMin - 15, 0));
                    image.Mutate(ctx => ctx.DrawText(labelText, font, Color.Red, textLocation));
                }
            }
        }

        // Save the annotated image.
        var outputPath = Path.Combine(Path.GetDirectoryName(imagePath)!, "annotated_output.png");
        image.Save(outputPath);
        Console.WriteLine($"Annotated image saved to: {outputPath}");
    }
}
