[![NuGet Version](https://img.shields.io/nuget/v/Tinn?label=NuGet)](https://www.nuget.org/packages/Tinn/)
[![NuGet Downloads](https://img.shields.io/nuget/dt/Tinn?label=Downloads)](https://www.nuget.org/packages/Tinn/)
[![Build](https://github.com/lawrence-laz/tinn-dotnet/workflows/Build/badge.svg)](https://github.com/lawrence-laz/tinn-dotnet/actions?query=workflow%3ABuild)

# Tinn (Tiny Neural Network)
Tinn is a tiny and dependency free neural network implementation for dotnet. The original library is written [by glouw in C](https://github.com/glouw/tinn).

# How to get started?
Download from [nuget.org](https://www.nuget.org/packages/Tinn/):
```powershell
PS> Install-Package Tinn
```

Create a neural network:
```csharp
var network = new TinyNeuralNetwork(inputCount: 2, hiddentCount: 4, outputCount: 1); 
```

Create or load your data set:
```csharp
// This is XOR operation example.
var input = new float[][]
{
    new []{ 1f, 1f }, // --> 0f
    new []{ 1f, 0f }, // --> 1f
    new []{ 0f, 1f }, // --> 1f
    new []{ 0f, 0f }, // --> 0f
};
var expected = new float[][]
{
    new []{ 0f }, // <-- 1f ^ 1f
    new []{ 1f }, // <-- 1f ^ 0f
    new []{ 1f }, // <-- 0f ^ 1f
    new []{ 0f }, // <-- 0f ^ 0f
};
```
Train the network until a desired accuracy is achieved:
```csharp
for (int i = 0; i < input.Length; i++)
{
    network.Train(input[i], expected[i], 1f);
}
// Note: you will probably have to loop this for a few times until network improves.
```
Try to predict some values:
```csharp
var prediction = network.Predict(new [] { 1f, 1f });  
// Will return probability close to 0f, since 1 ^ 1 = 0.
```
---

For concrete examples look in [example folder](https://github.com/lawrence-laz/tinn-dotnet/tree/main/example/) and [test folder](https://github.com/lawrence-laz/tinn-dotnet/tree/main/test/Tinn.Tests).