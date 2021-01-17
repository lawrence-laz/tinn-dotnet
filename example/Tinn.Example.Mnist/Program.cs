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
static void Shuffle(float[][] input, float[][] output)
{
    var random = new Random(0);

    for (int i = 0; i < input.Length; i++)
    {
        var j = random.Next(input.Length);
        (input[i], input[j]) = (input[j], input[i]);
        (output[i], output[j]) = (output[j], output[i]);
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
    );
var input = data.Select(x => x.Input).ToArray();
var expectedOutput = data.Select(x => x.Output).ToArray();
#endregion

#region Train neural network
var network = new TinyNeuralNetwork(inputCount, hiddentCount, outputCount);
var progress = new ProgressBar(learningIterations * input.Length, "Training...");

for (var i = 0; i < learningIterations; i++)
{
    for (int j = 0; j < input.Length; j++)
    {
        progress.Tick();
        network.Train(input[j], expectedOutput[j], learningRate);
    }
    Shuffle(input, expectedOutput);
    learningRate *= learningRateDecay;
}
#endregion

#region Test neural network
var predictedNumbers = input
    .Select(x => network.Predict(x))
    .Select(GetMaxIndex)
    .ToArray();

var actualNumbers = expectedOutput
    .Select(GetMaxIndex)
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
