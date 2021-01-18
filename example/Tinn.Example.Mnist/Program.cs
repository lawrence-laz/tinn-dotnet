using ShellProgressBar;
using System;
using System.IO;
using System.Linq;
using System.Net;
using Tinn;

const string datasetUri = "http://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.data";
const string dataSetFileName = "semeion.data";
const int inputCount = 256;
const int hiddentCount = 28;
const int outputCount = 10;

const int learningIterations = 10;
float learningRate = 1f;
const float learningRateDecay = 0.99f;

// Used for shuffling data set in between training iterations.
static void Shuffle<T>(T[] array)
{
    var random = new Random(0);

    for (int i = 0; i < array.Length; i++)
    {
        var j = random.Next(array.Length);
        (array[i], array[j]) = (array[j], array[i]);
    }
}

// Used for transforming categorical values to numeric.
static int GetMaxIndex(float[] values)
{
    var maxValue = float.MinValue;
    var maxValueIndex = -1;
    for (var i = 0; i < values.Length; i++)
    {
        if (values[i] >= maxValue)
        {
            maxValue = values[i];
            maxValueIndex = i;
        }
    }

    return maxValueIndex;
}

#region Get dataset
if (!File.Exists(dataSetFileName))
{
    Console.WriteLine("Downloading MNIST dataset...");
    using var webClient = new WebClient();
    webClient.DownloadFile(datasetUri, dataSetFileName);
    Console.WriteLine("Download completed.");
}
#endregion

#region Read dataset
var data = File.ReadAllLines(dataSetFileName)
    .Select(line => line.Split(" ").Select(float.Parse))
    .Select(number => (
        Input: number.Take(inputCount).ToArray(),
        Output: number.Skip(inputCount).Take(outputCount).ToArray())
    )
    .ToArray();
#endregion

#region Train neural network
var network = new TinyNeuralNetwork(inputCount, hiddentCount, outputCount);
var progress = new ProgressBar(learningIterations * data.Length, "Training...");

for (var i = 0; i < learningIterations; i++)
{
    foreach (var record in data)
    {
        progress.Tick();
        network.Train(record.Input, record.Output, learningRate);
    }

    Shuffle(data);
    learningRate *= learningRateDecay;
}
#endregion

#region Test neural network
var predictedNumbers = data
    .Select(x => network.Predict(x.Input))
    .Select(GetMaxIndex)
    .ToArray();

var actualNumbers = data
    .Select(record => GetMaxIndex(record.Output))
    .ToArray();

var correctlyGuessed = 0;
for (int i = 0; i < predictedNumbers.Length; i++)
{
    if (predictedNumbers[i] == actualNumbers[i])
    {
        correctlyGuessed++;
    }
}
var accuracy = (float)correctlyGuessed / actualNumbers.Length;
Console.Clear();
Console.WriteLine($"Achieved {accuracy:P2} accuracy.");
#endregion
