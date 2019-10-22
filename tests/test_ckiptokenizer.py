import os
import unittest
import logging

from unittest.mock import patch

import pytest

logger = logging.getLogger(__name__)

TEST_ROOT = os.path.dirname(os.path.abspath(__file__))


@patch("rukip.tokenizer.ckip_tokenizer.WS")
def test_ckip_tokenizer(mock_WS_class):
    user_dict_path = os.path.join(TEST_ROOT, "assets", "userdict.txt")
    component_config = {
        "model_path": "./data",
        "recommend_dict_path": user_dict_path,
        "coerce_dict_path": user_dict_path
    }

    from rukip.tokenizer import CKIPTokenizer
    ckip_tokenizer = CKIPTokenizer(component_config)
    expected_token_list = [
        ['昨天', '晚上', '吃', '了', '牛肉', '燴飯', '花', '了', '120', '元'],
    ]
    mock_WS_inst = mock_WS_class.return_value
    mock_WS_inst.return_value = expected_token_list

    tokens = ckip_tokenizer.tokenize("昨天晚上吃了牛肉燴飯花了120元")
    assert [t.text for t in tokens] == \
           ['昨天', '晚上', '吃', '了', '牛肉', '燴飯', '花', '了', '120',
            '元']
    assert [t.offset for t in tokens] == \
           [0, 2, 4, 5, 6, 8, 10, 11, 12, 15]


def test_ckip_tokenizer_wo_model():
    from rukip.tokenizer import CKIPTokenizer
    with pytest.raises(Exception):
        ckip_tokenizer = CKIPTokenizer()


def test_ckip_tokenizer_load_userdict():
    user_dict_path = os.path.join(TEST_ROOT, "assets", "userdict.txt")
    from rukip.tokenizer import CKIPTokenizer
    word_to_weigth = CKIPTokenizer.load_userdict(user_dict_path)
    assert word_to_weigth == {'土地公': '1', '土地婆': '1',
                              '公有': '2', '來亂的': '1',
                              '緯來體育台': '1'}
