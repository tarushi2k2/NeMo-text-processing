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

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_CHAR, GraphFst, delete_space


class MoneyFst(GraphFst):
    """
    Finite state transducer for verbalizing money, e.g.
        money { integer_part: "12" fractional_part: "05" currency: "$" } -> $12.05

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
        graph = unit + delete_space + decimal.numbers
        # graph |= unit + delete_space + cardinal.numbers
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()


from nemo_text_processing.inverse_text_normalization.hi.utils import apply_fst
from nemo_text_processing.inverse_text_normalization.hi.verbalizers.cardinal import CardinalFst
from nemo_text_processing.inverse_text_normalization.hi.verbalizers.decimal import DecimalFst

cardinal = CardinalFst()
decimal = DecimalFst()
money = MoneyFst(cardinal, decimal)
# input_text = 'money { integer_part: "९६" currency: "P"  }'
input_text = 'money { integer: "२०६" currency: "₹"  }'
# input_text = 'money { integer: "२०६" currency: "ლარი"  }'
output = apply_fst(input_text, money.fst)
print(output)
