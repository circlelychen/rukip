import os
import unittest
import logging
import ujson

from rukip.tokenizer import CKIPTokenizer


class CKIPTokenizerTestCase(unittest.TestCase):
    def setUp(self):
        self._logger = logging.getLogger(__name__)
        self._ckip_tokenizer = CKIPTokenizer()

    def tearDown(self):
        pass

    def test_tokenizer_s1(self):
        expected_resp_dict = {
            "tokens": [
                {'word': '昨天', 'tag': 'Nd'},
                {'word': '晚上', 'tag': 'Nd'},
                {'word': '吃', 'tag': 'VC'},
                {'word': '了', 'tag': 'Di'},
                {'word': '牛肉', 'tag': 'Na'},
                {'word': '燴飯', 'tag': 'Na'},
                {'word': '花', 'tag': 'VC'},
                {'word': '了', 'tag': 'Di'},
                {'word': '120', 'tag': 'Neu'},
                {'word': '元', 'tag': 'Nf'}
            ]
        }

        tokens = self._ckip_tokenizer.tokenize("昨天晚上吃了牛肉燴飯花了120元")
        assert [t.text for t in tokens] == \
               ['昨天', '晚上', '吃', '了', '牛肉', '燴飯', '花', '了', '120',
                '元']
        assert [t.offset for t in tokens] == \
               [0, 2, 4, 5, 6, 8, 10, 11, 12, 15]
        # assert [t.get("ckip_pos_tag") for t in tokens] == \
        #        ['Nd', 'Nd', 'VC', 'Di', 'Na', 'Na', 'VC', 'Di', 'Neu', 'Nf']
