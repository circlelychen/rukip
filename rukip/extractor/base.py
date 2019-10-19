import logging
import os
import typing
from typing import Any, Dict, List, Optional, Text, Tuple
import numpy as np

from rasa.nlu.config import InvalidConfigError, RasaNLUModelConfig
from rasa.nlu.extractors import EntityExtractor
from rasa.nlu.model import Metadata
from rasa.nlu.training_data import Message, TrainingData

from rasa.nlu.extractors import EntityExtractor

logger = logging.getLogger(__name__)


class BIOLUEntityExtractor(EntityExtractor):

    @staticmethod
    def _bilou_tags_from_offsets(tokens, entities, missing="O"):
        starts = {token.offset: i for i, token in enumerate(tokens)}
        ends = {token.end: i for i, token in enumerate(tokens)}
        bilou = ["-" for _ in tokens]
        # Handle entity cases
        for start_char, end_char, label in entities:
            start_token = starts.get(start_char)
            end_token = ends.get(end_char)
            # Only interested if the tokenization is correct
            if start_token is not None and end_token is not None:
                if start_token == end_token:
                    bilou[start_token] = "U-%s" % label
                else:
                    bilou[start_token] = "B-%s" % label
                    for i in range(start_token + 1, end_token):
                        bilou[i] = "I-%s" % label
                    bilou[end_token] = "L-%s" % label
        # Now distinguish the O cases from ones where we miss the tokenization
        entity_chars = set()
        for start_char, end_char, label in entities:
            for i in range(start_char, end_char):
                entity_chars.add(i)
        for n, token in enumerate(tokens):
            for i in range(token.offset, token.end):
                if i in entity_chars:
                    break
            else:
                bilou[n] = missing

        return bilou

    def most_likely_entity(self, idx, entities):
        if len(entities) > idx:
            entity_probs = entities[idx]
        else:
            entity_probs = None
        if entity_probs:
            label = max(entity_probs, key=lambda key: entity_probs[key])
            # if we are using bilou flags, we will combine the prob
            # of the B, I, L and U tags for an entity (so if we have a
            # score of 60% for `B-address` and 40% and 30%
            # for `I-address`, we will return 70%)
            return (
                label,
                sum([v for k, v in entity_probs.items() if k[2:] == label[2:]]),
            )
        else:
            return "", 0.0

    def _create_entity_dict(self, tokens, start, end, entity, confidence):
        _start = tokens[start].offset
        _end = tokens[end].end
        value = "".join(t.text for t in tokens[start: end + 1])

        return {
            "start": _start,
            "end": _end,
            "value": value,
            "entity": entity,
            "confidence": confidence,
        }

    @staticmethod
    def _entity_from_biolu_label(label):
        return label[2:]

    @staticmethod
    def _bilou_from_label(label):
        if len(label) >= 2 and label[1] == "-":
            return label[0].upper()
        return None

    def _find_bilou_end(self, word_idx, entities):
        ent_word_idx = word_idx + 1
        finished = False

        # get information about the first word, tagged with `B-...`
        label, confidence = self.most_likely_entity(word_idx, entities)
        entity_label = self._entity_from_biolu_label(label)

        while not finished:
            label, label_confidence = self.most_likely_entity(
                ent_word_idx, entities)

            confidence = min(confidence, label_confidence)

            if label[2:] != entity_label:
                # words are not tagged the same entity class
                logger.debug(
                    "Inconsistent BILOU tagging found, B- tag, L- "
                    "tag pair encloses multiple entity classes.i.e. "
                    "[B-a, I-b, L-a] instead of [B-a, I-a, L-a].\n"
                    "Assuming B- class is correct."
                )

            if label.startswith("L-"):
                # end of the entity
                finished = True
            elif label.startswith("I-"):
                # middle part of the entity
                ent_word_idx += 1
            else:
                # entity not closed by an L- tag
                finished = True
                ent_word_idx -= 1
                logger.debug(
                    "Inconsistent BILOU tagging found, B- tag not "
                    "closed by L- tag, i.e [B-a, I-a, O] instead of "
                    "[B-a, L-a, O].\nAssuming last tag is L-"
                )
        return ent_word_idx, confidence

    def _handle_bilou_label(self, word_idx, entities):
        label, confidence = self.most_likely_entity(word_idx, entities)
        entity_label = self._entity_from_biolu_label(label)

        if self._bilou_from_label(label) == "U":
            return word_idx, confidence, entity_label

        elif self._bilou_from_label(label) == "B":
            # start of multi word-entity need to represent whole extent
            ent_word_idx, confidence = self._find_bilou_end(word_idx, entities)
            return ent_word_idx, confidence, entity_label

        else:
            return None, None, None

    def _convert_bilou_tagging_to_entity_result(self, tokens, entities):
        # using the BILOU tagging scheme
        json_ents = []
        word_idx = 0
        while word_idx < len(tokens):
            end_idx, confidence, entity_label = self._handle_bilou_label(
                word_idx, entities
            )

            if end_idx is not None:
                ent = self._create_entity_dict(
                    tokens, word_idx, end_idx, entity_label, confidence
                )
                json_ents.append(ent)
                word_idx = end_idx + 1
            else:
                word_idx += 1
        return json_ents
