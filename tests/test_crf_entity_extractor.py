# coding=utf-8
import pytest

from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.training_data import TrainingData, Message
from rasa.nlu.tokenizers import Token

from rukip.extractor import CRFEntityExtractor


@pytest.fixture(scope="session")
def ner_crf_pos_feature_config():
    return {
        "features": [
            ["ckip_pos_tag"],
            ["ckip_pos_tag"],
            ["ckip_pos_tag"]
        ]
    }


def test_crf_extractor(ner_crf_pos_feature_config):

    ext = CRFEntityExtractor(component_config=ner_crf_pos_feature_config)
    m1 = Message("5000台幣等值多少日幣",
                 {
                     "intent": "匯率兌換",
                     "entities": [
                         {"start": 0, "end": 4, "value": "5000", "entity":
                          "money"},
                         {"start": 4, "end": 6, "value": "台幣", "entity":
                          "currency"},
                         {"start": 10, "end": 12, "value": "日幣", "entity":
                          "currency"}
                     ]
                 })
    m1.set("tokens", [
        Token("5000", 0, {"ckip_pos_tag": "Neu"}),
        Token("台幣", 4, {"ckip_pos_tag": "Na"}),
        Token("等值", 6, {"ckip_pos_tag": "VH"}),
        Token("多少", 8, {"ckip_pos_tag": "Neqa"}),
        Token("日幣", 10, {"ckip_pos_tag": "Na"})
    ])
    m2 = Message("兩千台幣可以換多少日圓",
                 {
                     "intent": "匯率兌換",
                     "entities": [
                         {
                             "start": 0,
                             "end": 2,
                             "value": "兩千",
                             "entity": "money"
                         },
                         {
                             "start": 2,
                             "end": 4,
                             "value": "台幣",
                             "entity": "currency"
                         },
                         {
                             "start": 9,
                             "end": 11,
                             "value": "日圓",
                             "entity": "currency"
                         }
                     ]
                 })
    m2.set("tokens", [
        Token("兩千", 0, {"ckip_pos_tag": "Neu"}),
        Token("台幣", 2, {"ckip_pos_tag": "Na"}),
        Token("可以", 4, {"ckip_pos_tag": "D"}),
        Token("換", 6, {"ckip_pos_tag": "VC"}),
        Token("多少", 7, {"ckip_pos_tag": "Neqa"}),
        Token("日圓", 9, {"ckip_pos_tag": "Nf"})
    ])
    examples = [m1, m2]

    # uses BILOU and the default features
    ext.train(TrainingData(training_examples=examples), RasaNLUModelConfig())

    crf_format = ext._from_text_to_crf(m1)
    assert [word[0] for word in crf_format] == [
        "5000", "台幣", "等值", "多少", "日幣"]
    assert [word[1] for word in crf_format] == [
        "Neu", "Na", "VH", "Neqa", "Na"]

    feats = ext._sentence_to_features(crf_format)
    assert "BOS" in feats[0]
    assert "EOS" in feats[-1]
    assert feats[1]["-1:ckip_pos_tag"] == "Neu"
    assert feats[1]["0:ckip_pos_tag"] == "Na"
    assert feats[1]["1:ckip_pos_tag"] == "VH"

    filtered = ext.filter_trainable_entities(examples)
    assert filtered[0].get("entities") == [
        {"start": 0, "end": 4, "value": "5000", "entity": "money"},
        {"start": 4, "end": 6, "value": "台幣", "entity": "currency"},
        {"start": 10, "end": 12, "value": "日幣", "entity": "currency"}
    ]

    extracted = ext.extract_entities(m2)
    assert [ent["entity"] for ent in extracted] == [
        "money", "currency", "currency"
    ]


def test_crf_extractor_misaligned(ner_crf_pos_feature_config):

    ext = CRFEntityExtractor(component_config=ner_crf_pos_feature_config)
    m1 = Message("目前存活存划算嗎",
                 {
                     "intent": "查詢存款利率",
                     "entities": [
                         {"start": 3, "end": 5, "value": "活存", "entity":
                          "acnt_type"}
                     ]
                 })
    m1.set("tokens", [
        Token("目前", 0, {"ckip_pos_tag": "Nd"}),
        Token("存活", 4, {"ckip_pos_tag": "VH"}),
        Token("存", 6, {"ckip_pos_tag": "VC"}),
        Token("划算", 8, {"ckip_pos_tag": "VH"}),
        Token("嗎", 10, {"ckip_pos_tag": "T"})
    ])
    examples = [m1]

    # uses BILOU and the default features
    ext.train(TrainingData(training_examples=examples), RasaNLUModelConfig())
    extracted = ext.extract_entities(m1)
    # import logging
    # logger = logging.getLogger(__name__)
    # logger.debug(extracted)
    assert extracted == []
