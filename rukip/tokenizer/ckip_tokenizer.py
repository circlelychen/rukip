from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os
import shutil
import typing
from typing import Any, Dict, List, Optional, Text

from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
from rasa.nlu.training_data import Message, TrainingData

from ckiptagger import construct_dictionary,  WS

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from rasa.nlu.model import Metadata


class CKIPTokenizer(Tokenizer, Component):

    provides = ["tokens"]

    language_list = ["zh"]

    name = "ckiptagger_tokenizer"

    defaults = {
        "use_cls_token": False,
        "model_path": None,
        "recommend_dict_path": {},
        "coerce_dict_path": {}
    }

    def __init__(self, component_config: Dict[Text, Any] = None) -> None:
        super(CKIPTokenizer, self).__init__(component_config)

        # must configure 'model_apth', or raise exception
        if not self.component_config.get("model_path"):
            raise Exception("model_path must be configured")

        # construct recommend_dict if 'recommend_dict' is  configured
        self._recommend_dict = {}
        if self.component_config.get("recommend_dict_path", None):
            self._recommend_dict = construct_dictionary(
                self.load_userdict(
                    self.component_config.get("recommend_dict_path")))

        # construct coerce_dict if 'coerce_dict' is  configured
        self._coerce_dict = {}
        if self.component_config.get("coerce_dict_path", None):
            self._coerce_dict = construct_dictionary(
                self.load_userdict(
                    self.component_config.get("coerce_dict_path")))

        self._ws = WS(
            self.component_config.get("model_path")
        )

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["ckiptagger"]

    @staticmethod
    def load_userdict(path: Text) -> Dict:
        word_to_weigth = {}
        with open(path, "rb") as fin:
            for lineno, ln in enumerate(fin, 1):
                line = ln.strip()
                if not isinstance(line, Text):
                    try:
                        line = line.decode('utf-8').lstrip('\ufeff')
                    except UnicodeDecodeError:
                        raise ValueError(
                            'dictionary file %s must be utf-8' % path)
                if not line:
                    continue
                line = line.strip()
                word, freq = line.split(' ')[:2]
                word_to_weigth[word] = freq
        return word_to_weigth

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUModelConfig, **Any) -> None

        for example in training_data.training_examples:
            example.set("tokens", self.tokenize(example.text))

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        message.set("tokens", self.tokenize(message.text))

    def tokenize(self, text):
        ckip_tokens = self._ws(
            [text],
            recommend_dictionary=self._recommend_dict,
            coerce_dictionary=self._coerce_dict)

        running_offset = 0
        tokens = []
        for word in ckip_tokens[0]:
            try:
                word_offset = text.index(word, running_offset)
            except ValueError as e:
                warnings.warn(
                    "ValueError on word: {0} on text: {1}".format(word, text))
                continue
            word_len = len(word)
            running_offset = word_offset + word_len
            token = Token(word, word_offset)
            tokens.append(token)
        return tokens
