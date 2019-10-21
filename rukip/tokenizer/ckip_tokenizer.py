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
from rasa.nlu.tokenizers import Token, Tokenizer
from rasa.nlu.training_data import Message, TrainingData

from ckiptagger import WS

from rasa.nlu.constants import (
    MESSAGE_RESPONSE_ATTRIBUTE,
    MESSAGE_INTENT_ATTRIBUTE,
    MESSAGE_TEXT_ATTRIBUTE,
    MESSAGE_TOKENS_NAMES,
    MESSAGE_ATTRIBUTES,
    MESSAGE_SPACY_FEATURES_NAMES,
    MESSAGE_VECTOR_FEATURE_NAMES,
)

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from rasa.nlu.model import Metadata


class CKIPTokenizer(Tokenizer, Component):

    provides = ["tokens"]

    language_list = ["zh"]

    name = "ckiptagger_tokenizer"

    defaults = {
        "model_path": None,
        "recommend_dict": None,
        "coerce_dict": None
    }

    def __init__(self, component_config: Dict[Text, Any] = None) -> None:
        super(CKIPTokenizer, self).__init__(component_config)
        if not self.component_config.get("model_path"):
            raise Exception("model_path must be configured")
        self._ws = WS(self.component_config.get("model_path"))

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["ckiptagger"]

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUModelConfig, **Any) -> None

        for example in training_data.training_examples:
            example.set("tokens", self.tokenize(example.text))

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        message.set("tokens", self.tokenize(message.text))

    def tokenize(self, text):
        ckip_tokens = self._ws([text])

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
