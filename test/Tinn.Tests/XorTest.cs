using FluentAssertions;
using System;
using System.Linq;
using Xunit;

namespace Tinn.Tests
{
    public class XorTest
    {
        [Fact]
        public void Given_a_xor_truth_table_neural_network_should_learn_the_xor_operation()
        {
            // Arrange
            var input = new float[][]
            {
                new []{ 1f, 1f },
                new []{ 1f, 0f },
                new []{ 0f, 1f },
                new []{ 0f, 0f },
            };
            var expected = new float[][]
            {
                new []{ 0f },
                new []{ 1f },
                new []{ 1f },
                new []{ 0f },
            };

            var network = new TinyNeuralNetwork(2, 4, 1);
            var learningRate = 1f;

            for (int i = 0; i < 1000; i++)
            {
                for (int j = 0; j < input.Length; j++)
                {
                    network.Train(input[j], expected[j], learningRate);
                }
                Shuffle(input, expected);
                learningRate *= .99f;
            }

            // Act
            var actualCategorical = input.Select(network.Predict);

            // Assert
            var actualNumeric = actualCategorical.Select(x => Math.Round(x[0]));
            var expectedNumeric = expected.Select(x => x[0]);
            actualNumeric.Should().BeEquivalentTo(expectedNumeric);
        }

        private static void Shuffle(float[][] input, float[][] output)
        {
            var random = new Random(0);

            for (int i = 0; i < input.Length; i++)
            {
                var j = random.Next(input.Length);
                (input[i], input[j]) = (input[j], input[i]);
                (output[i], output[j]) = (output[j], output[i]);
            }
        }
    }
}
