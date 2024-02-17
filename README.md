# Interaction Network Contextual Embedding (INCE)

Simple implementation of INCE, the algorithm described in [_"Graph Neural Network Contextual Embedding for
Deep Learning on Tabular Data"_](https://arxiv.org/abs/2303.06455)

## Model Description

INCE is a Deep Learning (DL) model for tabular data that employs Graph Neural Networks (GNNs) and, more specifically, 
Interaction Networks for contextual embedding.

First an encoder model
maps each tabular dataset feature into a latent vector or
embedding and then a decoder model takes the embeddings
and uses them to solve the supervised learning task.
The encoder model is composed by two components: the
columnar and the contextual embedding. The decoder model is
given by a Multi-Layer Perceptron (MLP) tuned to the learning
task to solve

<p align="center">
  <img src="https://github.com/MatteoSalvatori/INCE/blob/main/figs/encoder-decoder.png" alt="Encoder Decoder"/>
</p>

_COLUMNAR EMBEDDING:_ All features
(categorical and continuous) are individually projected
in a common dense latent space.

<p align="center">
  <img src="https://github.com/MatteoSalvatori/INCE/blob/main/figs/columnar-embedding.png" alt="Columnar Embedding"/>
</p>

_CONTEXTUAL EMBEDDING:_ The features obtained from columnar 
embedding are organized in a fully-connected graph with
an extra virtual node, called CLS as in BERT. Then,
a stack of Interaction Networks models the relationship
among all the nodes - original features and CLS virtual
node - and enhances their representation. The resulting
CLS virtual node is sent into the final classifier/regressor

<p align="center">
  <img src="https://github.com/MatteoSalvatori/INCE/blob/main/figs/contextual-embedding.PNG" alt="Contextual Embedding"/>
</p>

Schematic workflow of Interaction Network

<p align="center">
  <img src="https://github.com/MatteoSalvatori/INCE/blob/main/figs/ingnn.PNG" alt="IN GNN"/>
</p>

## Main Results

INCE has been tested on the benchmark described in the table below:

<table style="border-collapse: collapse; width: 100%; height: 108px;" border="1" align="center">
<tbody>
<tr style="height: 18px;">
<td style="width: 20%; height: 18px; text-align: center;"><strong>Dataset</strong></td>
<td style="width: 20%; height: 18px; text-align: center;"><strong>Rows</strong></td>
<td style="width: 20%; height: 18px; text-align: center;"><strong>Num. Feats</strong></td>
<td style="width: 20%; height: 18px; text-align: center;"><strong>Cat. Feats</strong></td>
<td style="width: 20%; height: 18px; text-align: center;"><strong>Task</strong></td>
</tr>
<tr style="height: 18px;">
<td style="width: 20%; height: 18px; text-align: center;">HELOC</td>
<td style="width: 20%; height: 18px; text-align: center;">9871</td>
<td style="width: 20%; height: 18px; text-align: center;">21</td>
<td style="width: 20%; height: 18px; text-align: center;">2</td>
<td style="width: 20%; height: 18px; text-align: center;">Binary</td>
</tr>
<tr style="height: 18px;">
<td style="width: 20%; height: 18px; text-align: center;">California Housing</td>
<td style="width: 20%; height: 18px; text-align: center;">20640</td>
<td style="width: 20%; height: 18px; text-align: center;">8</td>
<td style="width: 20%; height: 18px; text-align: center;">0</td>
<td style="width: 20%; height: 18px; text-align: center;">Regression</td>
</tr>
<tr style="height: 18px;">
<td style="width: 20%; height: 18px; text-align: center;">Adult Incoming</td>
<td style="width: 20%; height: 18px; text-align: center;">32561</td>
<td style="width: 20%; height: 18px; text-align: center;">6</td>
<td style="width: 20%; height: 18px; text-align: center;">8</td>
<td style="width: 20%; height: 18px; text-align: center;">Binary</td>
</tr>
<tr style="height: 18px;">
<td style="width: 20%; height: 18px; text-align: center;">Forest Cover Type</td>
<td style="width: 20%; height: 18px; text-align: center;">581 K</td>
<td style="width: 20%; height: 18px; text-align: center;">10</td>
<td style="width: 20%; height: 18px; text-align: center;">2 (4 + 40)</td>
<td style="width: 20%; height: 18px; text-align: center;">Multi-Class (7)</td>
</tr>
<tr style="height: 18px;">
<td style="width: 20%; height: 18px; text-align: center;">HIGGS</td>
<td style="width: 20%; height: 18px; text-align: center;">11 M</td>
<td style="width: 20%; height: 18px; text-align: center;">27</td>
<td style="width: 20%; height: 18px; text-align: center;">1</td>
<td style="width: 20%; height: 18px; text-align: center;">Binary</td>
</tr>
</tbody>
</table>

and compared with the following baselines: 

_Standard methods_: Linear Model, KNN, Decision Tree, Random
Forest, XGBoost, LightGBM, CatBoost. 

_Deep learning models_: MLP, DeepFM, DeepGBM, RLN, TabNet, 
VIME, TabTrasformer, NODE, Net-DNF, SAINT, FT-Transformer.

(See the paper for details and references)

The main results are summarized in table and plot shown below:

<table style="border-collapse: collapse; width: 100%; height: 144px;" border="1" align="center">
<tbody>
<tr style="height: 18px;">
<td style="width: 12.5%; height: 36px; text-align: center;" rowspan="2"><strong>Dataset</strong></td>
<td style="width: 12.5%; height: 36px; text-align: center;" rowspan="2"><strong>Metrics</strong></td>
<td style="width: 25%; height: 18px; text-align: center;" colspan="2"><strong>Best tree</strong></td>
<td style="width: 25%; height: 18px; text-align: center;" colspan="2"><strong>Best DL</strong></td>
<td style="width: 25%; height: 18px; text-align: center;" colspan="2"><strong>INCE</strong></td>
</tr>
<tr style="height: 18px;">
<td style="width: 12.5%; height: 18px; text-align: center;"><em>Result</em></td>
<td style="width: 12.5%; height: 18px; text-align: center;"><em>Model</em></td>
<td style="width: 12.5%; height: 18px; text-align: center;"><em>Result</em></td>
<td style="width: 12.5%; height: 18px; text-align: center;"><em>Model</em></td>
<td style="width: 12.5%; height: 18px; text-align: center;"><em>Result</em></td>
<td style="width: 12.5%; height: 18px; text-align: center;"><em>Rank</em></td>
</tr>
<tr style="height: 18px;">
<td style="width: 12.5%; height: 36px; text-align: center;">HELOC</td>
<td style="width: 12.5%; height: 36px; text-align: center;">Accuracy &uarr;</td>
<td style="width: 12.5%; height: 36px; text-align: center;">83.6 %</td>
<td style="width: 12.5%; height: 36px; text-align: center;">CatBoost</td>
<td style="width: 12.5%; height: 36px; text-align: center;">82.6 %</td>
<td style="width: 12.5%; height: 36px; text-align: center;">Net-DNF</td>
<td style="width: 12.5%; height: 36px; text-align: center;">84.2 &plusmn; 0.5 %</td>
<td style="width: 12.5%; height: 18px;">🥇 Abs.</td>
</tr>
<tr style="height: 18px;">
<td style="width: 12.5%; height: 36px; text-align: center;">California Housing</td>
<td style="width: 12.5%; height: 36px; text-align: center;">MSE &darr;</td>
<td style="width: 12.5%; height: 36px; text-align: center;">0.195</td>
<td style="width: 12.5%; height: 36px; text-align: center;">LightGBM</td>
<td style="width: 12.5%; height: 36px; text-align: center;">0.226</td>
<td style="width: 12.5%; height: 36px; text-align: center;">SAINT</td>
<td style="width: 12.5%; height: 36px; text-align: center;">0.216 &plusmn; 0.007</td>
<td style="width: 12.5%; height: 18px;">🥇 DL</td>
</tr>
<tr style="height: 18px;">
<td style="width: 12.5%; height: 36px; text-align: center;" rowspan="2">Adult Incoming</td>
<td style="width: 12.5%; height: 36px; text-align: center;" rowspan="2">Accuracy &uarr;</td>
<td style="width: 12.5%; height: 36px; text-align: center;" rowspan="2">87.4 %</td>
<td style="width: 12.5%; height: 36px; text-align: center;" rowspan="2">LightGBM</td>
<td style="width: 12.5%; height: 36px; text-align: center;" rowspan="2">86.1 %</td>
<td style="width: 12.5%; height: 36px; text-align: center;">DeepFM</td>
<td style="width: 12.5%; height: 36px; text-align: center;" rowspan="2">86.8 &plusmn; 0.3 %</td>
<td style="width: 12.5%; height: 18px;" rowspan="2">🥇 DL</td>
</tr>
<tr>
<td style="width: 12.5%; height: 36px; text-align: center;">SAINT</td>
</tr>
<tr style="height: 18px;">
<td style="width: 12.5%; height: 36px; text-align: center;" rowspan="2">Forest Cover Type</td>
<td style="width: 12.5%; height: 36px; text-align: center;" rowspan="2">Accuracy &uarr;</td>
<td style="width: 12.5%; height: 36px; text-align: center;" rowspan="2">97.3 %</td>
<td style="width: 12.5%; height: 36px; text-align: center;" rowspan="2">XGBoost</td>
<td style="width: 12.5%; height: 36px; text-align: center;" rowspan="2">96.3 %</td>
<td style="width: 12.5%; height: 36px; text-align: center;" rowspan="2">SAINT</td>
<td style="width: 12.5%; height: 36px; text-align: center;" rowspan="2">97.1 &plusmn; 0.1 %</td>
<td style="width: 12.5%; height: 18px;">🥇 DL</td>
</tr>
<tr style="height: 18px;">
<td style="width: 12.5%; height: 18px;">🥈 Abs.</td>
</tr>
<tr style="height: 18px;">
<td style="width: 12.5%; height: 36px; text-align: center;" rowspan="2">HIGGS</td>
<td style="width: 12.5%; height: 36px; text-align: center;" rowspan="2">Accuracy &uarr;</td>
<td style="width: 12.5%; height: 36px; text-align: center;" rowspan="2">77.6 %</td>
<td style="width: 12.5%; height: 36px; text-align: center;" rowspan="2">XGBoost</td>
<td style="width: 12.5%; height: 36px; text-align: center;" rowspan="2">79.8 %</td>
<td style="width: 12.5%; height: 36px; text-align: center;" rowspan="2">SAINT</td>
<td style="width: 12.5%; height: 36px; text-align: center;" rowspan="2">79.1 &plusmn; 0.0 %</td>
<td style="width: 12.5%; height: 18px;">🥈 DL</td>
</tr>
<tr style="height: 18px;">
<td style="width: 12.5%; height: 18px;">🥈 Abs.</td>
</tr>
</tbody>
</table>

<p align="center">
  <img src="https://github.com/MatteoSalvatori/INCE/blob/main/figs/boxplot_results.png" alt="Boxplot Results"/>
</p>


## How to use the code

Requirements: 
```bash
numpy==1.23.5
pandas==1.5.2
scikit-learn==1.1.3
torch==1.13.0+cu117
torch-cluster==1.6.0+pt113cu117
torch-geometric==2.2.0
torch-scatter==2.1.0+pt113cu117
torch-sparse==0.6.15+pt113cu117
torch-spline-conv==1.2.1+pt113cu117
tqdm==4.64.1
```

Train/Test INCE on California Housing dataset: 

```python
python main.py -d ./src/datasets/json_config/california_housing.json -m ./src/models/json_config/INCE.json
```


## Citation

If you use this codebase, please cite our work:
```bib
@article{villaizan2023graph,
    title="{Graph Neural Network contextual embedding for Deep Learning on Tabular Data}",
    author={Villaizán-Vallelado, Mario and Salvatori, Matteo and Carro Martinez, Belén and Sanchez Esguevillas, Antonio Javier},
    year={2024},
    journal={Neural Networks},
    url={https://arxiv.org/pdf/2303.06455.pdf}
}
```

