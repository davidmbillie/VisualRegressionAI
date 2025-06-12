using System;

class Program
{
    static void Main(string[] args)
    {
        string modelPath = "Models/resnet50-v2-7.onnx";
        string baselineImagePath = "Images/baseline.png";
        string currentImagePath = "Images/current.png";

        var comparer = new ImageComparer(modelPath);
        float similarity = comparer.CompareImages(baselineImagePath, currentImagePath);

        Console.WriteLine($"Similarity Score: {similarity:F3}");
        Console.WriteLine(similarity > 0.95 ? "Screens match." : "Screens differ!");
        Console.Read();
    }
}
