import logging
import typing
from typing import Any, Dict, List, Optional, Text

from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.featurizers import Featurizer
from rasa.nlu.training_data import Message, TrainingData

from ckiptagger import POS

logger = logging.getLogger(__name__)


class CKIPFeaturizer(Featurizer):

    name = "ckiptagger_featurizer"

    language_list = ["zh"]

    requires = ["tokens"]

    provides = ["ner_features"]

    defaults = {
        "model_path": None,
        "token_features": ["word", "pos"],
    }

    def __init__(self, component_config: Dict[Text, Any] = None) -> None:
        super(CKIPFeaturizer, self).__init__(component_config)

        # must configure 'model_path', or raise exception
        if not self.component_config.get("model_path"):
            raise Exception("model_path must be configured")

        self._pos = POS(
            self.component_config.get("model_path")
        )

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["ckiptagger"]

    def train(
        self, training_data: TrainingData, config: RasaNLUModelConfig, **kwargs: Any
    ) -> None:

        for example in training_data.intent_examples:
            example.set("ner_features", self.gen_ner_features(example))

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        message.set("ner_features", self.gen_ner_features(message))

    def gen_ner_features(self, message: Message):
        tokens = message.get("tokens")
        word_list = [token.text for token in tokens]
        if set(["word"]) == set(self.component_config["token_features"]):
            return [[word] for word in word_list]
        pos_list = self._pos([word_list])
        if set(["pos"]) == set(self.component_config["token_features"]):
            return [[pos] for pos in pos_list[0]]
        return [[word, pos] for pos, word in zip(pos_list[0], word_list)]
