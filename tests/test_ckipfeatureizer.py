import os
import unittest
import logging

from unittest.mock import patch

import pytest

from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.training_data import TrainingData, Message
from rasa.nlu.tokenizers import Token

logger = logging.getLogger(__name__)


@patch("rukip.featurizer.ckip_featurizer.POS")
def test_ckip_featurizer(mock_POS_class):
    expected_pos_list = [
        ['Nd', 'Nd', 'VC', 'Di', 'Na', 'Na', 'VC', 'Di', 'Neu', 'Nf']
    ]
    mock_POS_inst = mock_POS_class.return_value
    mock_POS_inst.return_value = expected_pos_list

    msg = Message.build(text="昨天晚上吃了牛肉燴飯花了120元",
                        intent="eat_dinner")
    msg.set("tokens", [
        Token("昨天", 0),
        Token("晚上", 2),
        Token("吃", 4),
        Token("了", 5),
        Token("牛肉", 6),
        Token("燴飯", 8),
        Token("花", 10),
        Token("了", 11),
        Token("120", 12),
        Token("元", 15)
    ])

    from rukip.featurizer import CKIPFeaturizer
    component_config = {
        "model_path": "./data"
    }

    ckip_featurizer = CKIPFeaturizer(component_config)
    ner_features = ckip_featurizer.gen_ner_features(msg)
    assert ner_features == [['昨天', 'Nd'], ['晚上', 'Nd'], ['吃', 'VC'],
                            ['了', 'Di'], ['牛肉', 'Na'], ['燴飯', 'Na'],
                            ['花', 'VC'], ['了', 'Di'], ['120', 'Neu'],
                            ['元', 'Nf']]

    component_config = {
        "model_path": "./data",
        "token_features": ["pos"]
    }
    ckip_featurizer = CKIPFeaturizer(component_config)
    ner_features = ckip_featurizer.gen_ner_features(msg)
    assert ner_features == [['Nd'], ['Nd'], ['VC'], ['Di'], ['Na'],
                            ['Na'], ['VC'], ['Di'], ['Neu'], ['Nf']]

    component_config = {
        "model_path": "./data",
        "token_features": ["word"]
    }
    ckip_featurizer = CKIPFeaturizer(component_config)
    ner_features = ckip_featurizer.gen_ner_features(msg)
    assert ner_features == [['昨天'], ['晚上'], ['吃'], ['了'], ['牛肉'],
                            ['燴飯'], ['花'], ['了'], ['120'], ['元']]
