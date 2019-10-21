import os
import unittest
import logging
import ujson

from unittest.mock import patch

import pytest

logger = logging.getLogger(__name__)


@patch("rukip.tokenizer.ckip_tokenizer.WS")
def test_ckip_tokenizer(mock_WS_class):
    component_config = {"model_path": "./data"}

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
