# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_CHAR,
    NEMO_NOT_QUOTE,
    GraphFst,
    delete_extra_space,
    delete_space,
    insert_space,
)


class MoneyFst(GraphFst):
    """
    Finite state transducer for verbalizing money, e.g.
        money { integer_part: "12" morphosyntactic_features: "," fractional_part: "05" currency: "$" } -> $12,05

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
        money = MoneyFst
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
        integer = (
            pynutil.delete("integer_part:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        fractional = (
            pynutil.insert(".")
            + pynutil.delete("fractional_part:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        graph = (
            integer
            + delete_space
            + pynini.closure(fractional + delete_space, 0, 1)
            + unit
            + delete_space
            + pynini.closure(insert_space + integer + delete_space + unit + delete_space, 0, 1)
        )

        # graph = unit + delete_space + integer + delete_space
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()


from nemo_text_processing.inverse_text_normalization.hi.utils import apply_fst
from nemo_text_processing.inverse_text_normalization.hi.verbalizers.cardinal import CardinalFst
from nemo_text_processing.inverse_text_normalization.hi.verbalizers.decimal import DecimalFst

cardinal = CardinalFst()
decimal = DecimalFst()
money = MoneyFst(cardinal, decimal)
# input_text = 'money { integer_part: "९६" currency: "P"  }'
# input_text = 'money { integer_part: "२०६" currency: "₹"  }'
# input_text = 'money { integer_part: "२०६" currency: "ლარი"  }'
input_text = 'money { integer_part: "१२०"  fractional_part: "९६" currency: "֏"  }'
# input_text = 'money { integer_part: "१२०१३"  fractional_part: "७७७" currency: "֏"  }'
# input_text = 'money { integer_part: "२०६" currency: "₹"   integer_part: "२०६" currency: "P"  }'
# input_text = 'money { integer_part: "५००" currency: "₹"   integer_part: "५०" currency: "P"  }'
output = apply_fst(input_text, money.fst)
print(output)
