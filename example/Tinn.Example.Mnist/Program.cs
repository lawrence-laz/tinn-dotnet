using System.Globalization;
using ShellProgressBar;
using Tinn;

const string datasetUri = "http://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.data";
const string dataSetFileName = "semeion.data";
const int inputCount = 256;
const int hiddenCount = 28;
const int outputCount = 10;

// reserve ~10% of data for verification
const int verifyCount = 150;

const int learningIterations = 10;
float learningRate = 0.1f;
const float learningRateDecay = 0.95f;
var random = new Random(0);

if (File.Exists(dataSetFileName) == false)
{
    Console.WriteLine("Downloading MNIST dataset...");
    using HttpClient client = new();
    using Stream file = File.Create(dataSetFileName);
    Stream stream = await client.GetStreamAsync(datasetUri);
    stream.CopyTo(file);
    Console.WriteLine("Download completed.");
}

(float[] Input, float[] Output)[] allData = File.ReadAllLines(dataSetFileName)
    .Select(line => line.Split(" ").Select(x => float.Parse(x, CultureInfo.InvariantCulture)))
    .Select(number => (
        Input: number.Take(inputCount).ToArray(),
        Output: number.Skip(inputCount).Take(outputCount).ToArray())
    )
    .ToArray();

(float[] Input, float[] Output)[] learningData = allData.Skip(verifyCount).ToArray();
(float[] Input, float[] Output)[] verifyData = allData.Take(verifyCount).ToArray();

var network = new TinyNeuralNetwork(inputCount, hiddenCount, outputCount);
var progress = new ProgressBar(learningIterations, "Training...");

string currentAccuracy = "";
for (var i = 0; i < learningIterations; i++)
{
    using ChildProgressBar child = progress.Spawn(learningData.Length, "iteration " + i, new ProgressBarOptions { CollapseWhenFinished = true });
    foreach ((float[] Input, float[] Output, int n) in learningData.Select(((float[] i, float[] o) data, int n) => (data.i, data.o, n)))
    {
        network.Train(Input, Output, learningRate);
        if (n == learningData.Length - 1)
        {
            currentAccuracy = ComputeAccuracy(verifyData, network);
        }

        child.Tick();
    }

    Shuffle(learningData);
    learningRate *= learningRateDecay;
    progress.Tick(currentAccuracy);
}

network.Save("network.tinn");
currentAccuracy = ComputeAccuracy(verifyData, network);
Console.WriteLine(currentAccuracy);

// Used for shuffling data set in between training iterations.
void Shuffle<T>(T[] array)
{
    for (int i = 0; i < array.Length; i++)
    {
        var j = random.Next(array.Length);
        (array[i], array[j]) = (array[j], array[i]);
    }
}

string ComputeAccuracy((float[] Input, float[] Output)[] subset, TinyNeuralNetwork network)
{
    int[] predictedNumbers = subset
        .Select(x => network.Predict(x.Input))
        .Select(f => f.Select((n, i) => (n, i)).Max().i)
        .ToArray();

    int[] actualNumbers = subset
        .Select(record => record.Output.Select((n, i) => (n, i)).Max().i)
        .ToArray();

    double correctlyGuessed = predictedNumbers.Zip(actualNumbers, (l, r) => l == r ? 1.0 : 0.0).Sum();
    double accuracy = correctlyGuessed / actualNumbers.Length;
    return $"Achieved {accuracy:P2} accuracy.";
}
