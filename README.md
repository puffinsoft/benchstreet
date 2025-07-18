<img src="docs/images/logo.png" width="600">

<hr/>

![](https://img.shields.io/badge/Keras-FF0000?style=for-the-badge&logo=keras&logoColor=white) ![](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white) ![](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white) ![](https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=WeightsAndBiases&logoColor=white) ![](https://img.shields.io/badge/-HuggingFace-FDEE21?style=for-the-badge&logo=HuggingFace&logoColor=black)

**Benchstreet** is a curated collection of time series prediction models designed to help developers evaluate and compare the performance of different approaches in **one-shot**, **long-term** financial data forecasting.

The models are trained on 20 years of S&P 500 daily closing prices provided by [Investing.com](https://investing.com/indices/us-spx-500-historical-data).

> [!IMPORTANT]
> This is not an objective benchmark! It's intended as a qualitative guide and a reference on how to implement these models.

![](docs/images/diagram.png)

<hr />

### Table of Contents

| Model Type                           |                                                       |
|--------------------------------------|-------------------------------------------------------|
| Transformer/Foundation Models        | TimesFM ([baseline](#timesfm-baseline) • [**fine-tuned**](#timesfm-fine-tuned)), Chronos ([baseline](#chronos-baseline) • [**fine-tuned**](#chronos-fine-tuned)) |
| Feedforward Neural Networks (FNNs)   | MLP ([recursive](#mlp-recursive) • [vector](#mlp-vector)), N-BEATS ([direct](#n-beats)) |
| Convolutional Neural Networks (CNNs) | 1D-CNN ([recursive](#1d-cnn-recursive) • [vector](#1d-cnn-vector)), TemporalCN ([vector](#temporalcn)) |
| Recurrent Neural Networks (RNNs)     | LSTM ([recursive](#lstm-recursive) • [vector](#lstm-vector) • [encoder-decoder](#lstm-encoder-decoder)), GRU ([recursive](#gru-recursive) • [vector](#gru-vector)) |
| Statistical Models                   | ARIMA ([recursive](#arima)), SARIMAX ([vector](#sarimax)), FBProphet ([direct](#fbprophet)) |

Want a model added to this list? Raise an issue [here](https://github.com/puffinsoft/benchstreet/issues) or [make a PR](https://github.com/puffinsoft/benchstreet/pulls)!

> [!TIP]
> **The winner**: [N-BEATS](#n-beats). High accuracy with extremely low training time.

<hr />

### TimesFM

#### timesfm-baseline

<img src="docs/images/timesfm_baseline.png"/>

[`timesfm/baseline.py`](benchstreet/timesfm/baseline.py)

<details>
<summary>view magnified graph</summary>
<img src="docs/images/timesfm_baseline_ZOOMED.png" width="80%">
</details>

#### timesfm-fine-tuned

<img src="docs/images/timesfm_finetuned.png"/>

[`timesfm/fine_tune.ipynb`](benchstreet/timesfm/fine_tune.ipynb) • [download on huggingface 🤗](https://huggingface.co/ColonelParrot/benchstreet-timesfm-2.0-500m-torch-sp500)

<details>
<summary>view magnified graph</summary>
<img src="docs/images/timesfm_finetuned_ZOOMED.png" width="80%">
</details>

### Chronos

#### chronos-baseline

<img src="docs/images/chronos_baseline.png"/>

[`chronos/baseline.py`](benchstreet/chronos/baseline.py)

<details>
<summary>view magnified graph</summary>
<img src="docs/images/chronos_baseline_ZOOMED.png" width="80%">
</details>

#### chronos-fine-tuned

<img src="docs/images/chronos_finetuned.png"/>

[`chronos/fine_tune.ipynb`](benchstreet/chronos/fine_tune.ipynb) • [download on huggingface 🤗](https://huggingface.co/ColonelParrot/benchstreet-chronos-t5-small-sp500)

<details>
<summary>view magnified graph</summary>
<img src="docs/images/chronos_finetuned_ZOOMED.png" width="80%">
</details>

### MLP

#### mlp-recursive

<img src="docs/images/mlp_recursive.png"/>

[`mlp/recursive.py`](benchstreet/mlp/recursive.py)

<details>
<summary>view magnified graph</summary>
<img src="docs/images/mlp_recursive_ZOOMED.png" width="80%">
</details>

#### mlp-vector

<img src="docs/images/mlp_vector_multistep.png"/>

[`mlp/vector.py`](benchstreet/mlp/vector.py)

<details>
<summary>view magnified graph</summary>
<img src="docs/images/mlp_vector_multistep_ZOOMED.png" width="80%">
</details>

### N-BEATS

<img src="docs/images/nbeats.png"/>

[`n_beats/direct.py`](benchstreet/n_beats/direct.py)

<details>
<summary>view magnified graph</summary>
<img src="docs/images/nbeats_ZOOMED.png" width="80%">
</details>

### 1D-CNN

#### 1d-cnn-recursive

<img src="docs/images/cnn_1d_recursive.png"/>

[`cnn/recursive.py`](benchstreet/cnn/recursive.py)

<details>
<summary>view magnified graph</summary>
<img src="docs/images/cnn_1d_recursive_ZOOMED.png" width="80%">
</details>

#### 1d-cnn-vector

<img src="docs/images/cnn_1d_vector_multistep.png"/>

[`cnn/vector.py`](benchstreet/cnn/vector.py)

<details>
<summary>view magnified graph</summary>
<img src="docs/images/cnn_1d_vector_multistep_ZOOMED.png" width="80%">
</details>

### TemporalCN

<img src="docs/images/tcn_vector_multistep.png"/>

[`tcn/vector.py`](benchstreet/tcn/vector.py)

<details>
<summary>view magnified graph</summary>
<img src="docs/images/tcn_vector_multistep_ZOOMED.png" width="80%">
</details>

### LSTM

#### lstm-recursive

<img src="docs/images/lstm_recursive.png"/>

[`lstm/recursive.py`](benchstreet/lstm/recursive.py)

<details>
<summary>view magnified graph</summary>
<img src="docs/images/lstm_recursive_ZOOMED.png" width="80%">
</details>

#### lstm-vector

<img src="docs/images/lstm_vector_multistep.png"/>

[`lstm/vector.py`](benchstreet/lstm/vector.py)

<details>
<summary>view magnified graph</summary>
<img src="docs/images/lstm_vector_multistep_ZOOMED.png" width="80%">
</details>

#### lstm-encoder-decoder

<img src="docs/images/lstm_encdec_multistep.png"/>

[`lstm/encdec.py`](benchstreet/lstm/encdec.py)

<details>
<summary>view magnified graph</summary>
<img src="docs/images/lstm_encdec_multistep_ZOOMED.png" width="80%">
</details>

### GRU

#### gru-recursive

<img src="docs/images/gru_recursive.png"/>

[`gru/recursive.py`](benchstreet/gru/recursive.py)

<details>
<summary>view magnified graph</summary>
<img src="docs/images/gru_recursive_ZOOMED.png" width="80%">
</details>

#### gru-vector

<img src="docs/images/gru_vector_multistep.png"/>

[`gru/vector.py`](benchstreet/gru/vector.py)

<details>
<summary>view magnified graph</summary>
<img src="docs/images/gru_vector_multistep_ZOOMED.png" width="80%">
</details>

### ARIMA

<img src="docs/images/arima_recursive.png"/>

[`arima/recursive.py`](benchstreet/arima/recursive.py)

<details>
<summary>view magnified graph</summary>
<img src="docs/images/arima_recursive_ZOOMED.png" width="80%">
</details>

### SARIMAX

<img src="docs/images/sarima.png"/>

[`sarimax/direct.ipynb`](benchstreet/sarimax/direct.ipynb)

<details>
<summary>view magnified graph</summary>
<img src="docs/images/sarima_ZOOMED.png" width="80%">
</details>

### FBProphet

<img src="docs/images/prophet.png"/>

[`fbprophet/direct.py`](benchstreet/fbprophet/direct.py)

<details>
<summary>view magnified graph</summary>
<img src="docs/images/prophet_ZOOMED.png" width="80%">
</details>

<hr />

Want a model added to this list? Raise an issue [here](https://github.com/puffinsoft/benchstreet/issues) or [make a PR](https://github.com/puffinsoft/benchstreet/pulls)!