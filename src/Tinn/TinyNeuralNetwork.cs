using System.Globalization;

namespace Tinn;

/// <summary>
/// A tiny neural network with one hidden layer and configurable parameters.
/// </summary>
public class TinyNeuralNetwork
{
    internal float[] _weights;
    internal float[] _biases;
    internal float[] _hiddenLayer;
    internal float[] _outputLayer;
    internal int _inputCount;
    internal Random _random;

    /// <summary>
    /// Creates an instance of an untrained neural network.
    /// </summary>
    /// <param name="inputCount">Number of inputs or features.</param>
    /// <param name="hiddentCount">Number of hidden neurons in a hidden layer.</param>
    /// <param name="outputCount">Number of outputs or classes.</param>
    /// <param name="seed">A seed for random generator to produce predictable results.</param>
    public TinyNeuralNetwork(int inputCount, int hiddentCount, int outputCount, int seed = default)
    {
        _random = new Random(seed);
        _inputCount = inputCount;
        _weights = Enumerable.Range(0, hiddentCount * (inputCount + outputCount)).Select(_ => (float)_random.NextDouble() - 0.5f).ToArray();
        _biases = Enumerable.Range(0, 2).Select(_ => (float)_random.NextDouble() - 0.5f).ToArray(); // Tinn only supports one hidden layer so there are two biases.
        _hiddenLayer = new float[hiddentCount];
        _outputLayer = new float[outputCount];
    }

    private TinyNeuralNetwork(float[] weights, float[] biases, float[] hiddenLayer, float[] outputLayer, int inputCount, int seed)
    {
        _weights = weights;
        _biases = biases;
        _hiddenLayer = hiddenLayer;
        _outputLayer = outputLayer;
        _inputCount = inputCount;
        _random = new Random(seed);
    }

    /// <summary>
    /// Loads a pretrained neural network from a `*.tinn` file.
    /// </summary>
    /// <param name="path">An absolute or a relative path to the `*.tinn` file.</param>
    /// <param name="seed">A seed for random generator to produce predictable results.</param>
    /// <returns>An instance of a pretrained <see cref="TinyNeuralNetwork"/>.</returns>
    public static TinyNeuralNetwork Load(string path, int seed = default)
    {
        using StreamReader reader = new(path);
        string metaData = ReadLine();
        var counts = metaData.Split(' ').Select(int.Parse).ToArray();
        var inputCount = counts[0];
        var hiddenCount = counts[1];
        var outputCount = counts[2];

        var weights = new float[hiddenCount * (inputCount + outputCount)];
        var biases = new float[2];
        var hiddenLayer = new float[hiddenCount];
        var outputLayer = new float[outputCount];
        var biasCount = 2;
        for (var i = 0; i < biasCount; i++)
        {
            biases[i] = float.Parse(ReadLine(), CultureInfo.InvariantCulture);
        }

        for (int i = 0; i < weights.Length; i++)
        {
            weights[i] = float.Parse(ReadLine(), CultureInfo.InvariantCulture);
        }

        TinyNeuralNetwork network = new(weights, biases, hiddenLayer, outputLayer, inputCount, seed);
        return network;

        string ReadLine()
        {
            return reader.ReadLine() ?? throw new Exception("invalid file");
        }
    }

    /// <summary>
    /// Predicts outputs from a given input.
    /// </summary>
    /// <param name="input">A float array matching the length of input count.</param>
    /// <returns>An array of predicted probabilities for each class. </returns>
    public float[] Predict(float[] input)
    {
        PropagateForward(input);
        return _outputLayer;
    }

    /// <summary>
    /// Trains neural network on a single data record.
    /// </summary>
    /// <param name="input">Records input or feature values.</param>
    /// <param name="expectedOutput">Actual record's class in a categorical format.</param>
    /// <param name="learningRate">Learning rate of a training.</param>
    public void Train(float[] input, float[] expectedOutput, float learningRate)
    {
        PropagateForward(input);
        PropagateBackward(input, expectedOutput, learningRate);
    }

    /// <summary>
    /// Saves a trained neural network to a `*.tinn` file.
    /// </summary>
    /// <param name="path">An absolute or a relative path to the `*.tinn` file.</param>
    public void Save(string path)
    {
        using StreamWriter writer = new FormattingStreamWriter(path, CultureInfo.InvariantCulture);
        writer.WriteLine($"{_inputCount} {_hiddenLayer.Length} {_outputLayer.Length}");
        foreach (float bias in _biases)
            writer.WriteLine(bias);

        foreach (float weight in _weights)
            writer.WriteLine(weight);
    }

    /// <summary>
    /// Get total error
    /// </summary>
    /// <param name="expectedOutput">Actual record's class in a categorical format.</param>
    /// <returns>Aggregated error value indicating how far off the neural network is on the training data set.</returns>
    public float GetTotalError(float[] expectedOutput)
    {
        return GetTotalError(expectedOutput, _outputLayer);
    }

    private void PropagateForward(float[] input)
    {
        // Calculate hidden layer neuron values.
        for (var i = 0; i < _hiddenLayer.Length; i++)
        {
            var sum = 0.0f;
            for (var j = 0; j < _inputCount; j++)
                sum += input[j] * _weights[i * _inputCount + j];

            _hiddenLayer[i] = ActivationFunction(sum + _biases[0]);
        }

        // Calculate output layer neuron values.
        for (int i = 0; i < _outputLayer.Length; i++)
        {
            var sum = 0.0f;

            for (int j = 0; j < _hiddenLayer.Length; j++)
                sum += _hiddenLayer[j] * _weights[(_hiddenLayer.Length * _inputCount) + i * _hiddenLayer.Length + j];

            _outputLayer[i] = ActivationFunction(sum + _biases[1]);
        }
    }

    private void PropagateBackward(float[] input, float[] expectedOutput, float learningRate)
    {
        for (var i = 0; i < _hiddenLayer.Length; i++)
        {
            var sum = 0.0f;

            // Calculate total error change with respect to output.
            for (var j = 0; j < _outputLayer.Length; j++)
            {
                float a = LossFunctionPartialDerivative(_outputLayer[j], expectedOutput[j]);
                float b = ActivationFunctionPartialDerivative(_outputLayer[j]);
                sum += a * b * _weights[(_hiddenLayer.Length * _inputCount) + j * _hiddenLayer.Length + i];

                // Correct weights in hidden to output layer.
                _weights[(_hiddenLayer.Length * _inputCount) + j * _hiddenLayer.Length + i] -= learningRate * a * b * _hiddenLayer[i];
            }

            // Correct weights in input to hidden layer.
            for (int j = 0; j < _inputCount; j++)
            {
                _weights[i * _inputCount + j] -= learningRate * sum * ActivationFunctionPartialDerivative(_hiddenLayer[i]) * input[j];
            }
        }
    }

    private static float ActivationFunction(float value)
    {
        return 1.0f / (1.0f + (float)Math.Exp(-value));
    }

    private static float ActivationFunctionPartialDerivative(float value)
    {
        return value * (1f - value);
    }

    private static float LossFunction(float expected, float actual)
    {
        return 0.5f * (expected - actual) * (expected - actual);
    }

    private static float LossFunctionPartialDerivative(float actual, float expected)
    {
        return actual - expected;
    }

    private static float GetTotalError(float[] expected, float[] actual)
    {
        float totalError = expected.Zip(actual, (e, a) => LossFunction(e, a)).Sum();
        return totalError;
    }

    private class FormattingStreamWriter : StreamWriter
    {
        private readonly IFormatProvider _formatProvider;

        public FormattingStreamWriter(string path, IFormatProvider formatProvider)
            : base(path)
        {
            _formatProvider = formatProvider;
        }

        public override IFormatProvider FormatProvider => _formatProvider;
    }
}
