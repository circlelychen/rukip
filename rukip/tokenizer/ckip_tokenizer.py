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

    provides = [MESSAGE_TOKENS_NAMES[attribute]
                for attribute in MESSAGE_ATTRIBUTES]

    # provides = [
    #     MESSAGE_VECTOR_FEATURE_NAMES[attribute]
    #     for attribute in SPACY_FEATURIZABLE_ATTRIBUTES
    # ] + [MESSAGE_NER_FEATURES_ATTRIBUTE]

    language_list = ["zh"]

    name = "tokenizer_ckip"

    # defaults = {
    #     "ckip_pos_tag": True
    # }
    defaults = {
        "dictionary_path": None,
        # Flag to check whether to split intents
        "intent_tokenization_flag": False,
        # Symbol on which intent should be split
        "intent_split_symbol": "_",
    }  # default don't load custom dictionary

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUModelConfig, **Any) -> None

        for example in training_data.training_examples:
            example.set("tokens", self.tokenize(example.text))

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        message.set("tokens", self.tokenize(message.text))

    def tokenize(self, text):
        # type: (Text) -> List[Token]
        return None
