# rukip
An Embedded CKIP Rasa NLU Components

## Introduction
This open-source library implements [Rasa](https://github.com/RasaHQ/rasa) custom components. 

It offers```tokenizer``` powered by [ckiptagger](https://github.com/ckiplab/ckiptagger) as condadate in RasaNLU pipeline.

## Installation
This library is built on [python >= 3.6](https://www.python.org/downloads/release/python-367/)

```bash
pip install rukip
```
## Usage

### Download model files
The model files are available on several mirror sites.

- [iis-ckip](http://ckip.iis.sinica.edu.tw/data/ckiptagger/data.zip)
- [gdrive-ckip](https://drive.google.com/drive/folders/105IKCb88evUyLKlLondvDBoh7Dy_I1tm)
- [gdrive-jacobvsdanniel](https://drive.google.com/drive/folders/15BDjL2IaX3eYdFVzT422VwCb743Hrbi3)

You can download and extract to the desired path by the following steps:

1. Downloads to ```./data.zip``` 
2. Extracts to ```./data/```

Add ```CKIPTokenizer``` component into rasa nlu pipeline and configure model_path as ```./data```. 

The following is the example ***Rasa NLU config file*** 

```xml
language: "zh"
pipeline:
  - name: "rukip.tokenizer.CTBCTokenizer"
    model_path: "./data"
    recommend_dict_path: ""
    cooerce_dict_path: ""
  - name: "CountVectorsFeaturizer"
  - name: "EmbeddingIntentClassifier"
```

### Components
#### CKIPTokenizer
This component has **one** required field (model_path) to be configured and offers **two** optional fields for user to assign dictionaries.

* ```recommend_dict_path ``` is the file containing list of user-defined recommended-word
* ```cooerce_dict_path``` is the file containing a list of must-word. 

The following is the example of user-defined dictionary. Each line shows one pair of word and weight.

```
土地公 1
土地婆 1
公有 2
來亂的 1
緯來體育台 1

``` 

## Development
```bash
$> git clone git@github.com:circlelychen/rukip.git
$> pip install -r requirements-to-freeze.dev.txt
$> make test
```

## License
licensed under the GNU General Public License v3.0
