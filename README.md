
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>

  <h1>LS-PLL Algorithm Implementation</h1>

  <p>This repository contains the implementation of the LS-PLL (Label Smoothing for Partial Labelled dataset) algorithm.</p>

  <h2>Usage</h2>

  <ol>
    <li>Run <code>preprocessing.py</code> to generate the dataset.</li>
    <li>Run <code>main.py</code> to calculate the training loss and testing accuracy.</li>
    <li>Run <code>TSEN.py</code> to generate the t-SNE plots using the model weights and parameters saved in the above step.</li>
  </ol>

  <h2>Instructions</h2>

  <p>Follow the steps below to use the LS-PLL algorithm:</p>

  <h3>Step 1: Generate Dataset</h3>

  <pre>
  <code>python preprocessing.py</code>
  </pre>

  <p>This command will generate the required numpy files for training and testing using four different standard datasets, i.e., CIFAR-10, CIFAR-100, Fashion-MNIST, and Kuzushiji-MNIST.</p>

  <h3>Step 2: Calculate Loss and Accuracy</h3>

  <pre>
  <code>python main_.py</code>
  </pre>

  <p>Execute this command to calculate the training loss and testing accuracy using the LS-PLL algorithm and save the model weights and parameters.<b> ResNet-18 </b> is used for CIFAR-10 and CIFAR-100 experimentation and <b> LeNet-5 </b> is used for Fashion-MNIST and Kuzushiji-MNIST experimentation. The input files are generated from Step 1</p>

  <h3>Step 3: Generate t-SNE plots</h3>

  <pre>
  <code>python TSEN.py</code>
  </pre>

  <p>Execute this command to generate the t-SNE plots. The trained model parameters are used from step 2 and the input files are used from Step 1</p>
 
  


</body>
</html>
