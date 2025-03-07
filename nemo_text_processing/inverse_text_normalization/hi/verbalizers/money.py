# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2024 and onwards Google, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pynini
from pynini.lib import pynutil
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_CHAR, GraphFst, delete_space, insert_space


class MoneyFst(GraphFst):
    """
    Finite state transducer for verbalizing money, e.g.
        बहत्तर लाइटकॉइन -> money { integer_part: "७२" currency: "ł" } -> ł७२

    Args:
        decimal: DecimalFst
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst):
        super().__init__(name="money", kind="verbalize")
        unit = (
            pynutil.delete("currency:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_CHAR - " ", 1)
            + pynutil.delete("\"")
        )
        graph = (
            unit
            + delete_space
            + decimal.numbers
            + delete_space
            + pynini.closure(insert_space + unit + delete_space + decimal.numbers + delete_space, 0, 1)
        )
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
