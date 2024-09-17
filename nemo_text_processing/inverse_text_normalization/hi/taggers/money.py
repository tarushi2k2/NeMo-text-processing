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

from nemo_text_processing.inverse_text_normalization.hi.graph_utils import (
    GraphFst,
    convert_space,
    delete_extra_space,
    delete_space,
    insert_space,
)
from nemo_text_processing.inverse_text_normalization.hi.utils import apply_fst, get_abs_path


class MoneyFst(GraphFst):
    """
    Finite state transducer for classifying measure
        e.g. ऋण बारह किलोग्राम -> money { integer_part: "१२" fractional_part: "५०" currency: "$" }
        e.g. ऋण बारह किलोग्राम -> money { integer_part: "१२" currency: "$" }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
        money: MoneyFst
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst):
        super().__init__(name="money", kind="classify")

        cardinal_graph = cardinal.graph_no_exception
        decimal_graph = decimal.final_graph_wo_negative
        currency_graph = pynini.string_file(get_abs_path("data/money/currency.tsv")).invert()

        self.integer = pynutil.insert("integer_part: \"") + cardinal_graph + pynutil.insert("\"")
        self.fraction = decimal_graph
        self.currency = pynutil.insert("currency: \"") + currency_graph + pynutil.insert("\" ")
        aur = pynutil.delete("और")

        graph_currency_decimal = self.fraction + delete_extra_space + self.currency
        graph_currency_cardinal = self.integer + delete_extra_space + self.currency
        graph_rupay_and_paisa = (
            graph_currency_cardinal
            + delete_extra_space
            + pynini.closure(aur + delete_extra_space, 0, 1)
            + graph_currency_cardinal
        )

        graph = graph_currency_decimal | graph_currency_cardinal | graph_rupay_and_paisa
        self.graph = graph.optimize()

        final_graph = self.add_tokens(graph)
        self.fst = final_graph


# from nemo_text_processing.inverse_text_normalization.hi.taggers.decimal import DecimalFst
# from nemo_text_processing.inverse_text_normalization.hi.taggers.cardinal import CardinalFst
# cardinal = CardinalFst()
# decimal = DecimalFst(cardinal)
# money = MoneyFst(cardinal, decimal)
# input_text = "सत्रह दशमलव दो नौ बहरीन दिरहम"
# input_text = "सत्रह दशमलव दो नौ बहामियन डॉलर"
# input_text = "बारह हज़ार तेरह दशमलव सात सात सात आर्मेनियाई ड्राम"
# input_text = "बारह हज़ार तेरह लाइबेरियन डॉलर"
# input_text = "बारह हज़ार वोन"
# input_text = "बहत्तर दशमलव आठ तीन सात लाइटकॉइन"
# input_text = "बहत्तर लाइटकॉइन"
# input_text = "दो सौ छह लारी"
# input_text = "दो सौ छह रुपये" #बहत्तर पैसे"
# input_text = "दो सौ छह रुपये दो सौ छह पैसे"
# input_text = "दो सौ छह रुपये और दो सौ छह पैसे"
# input_text = "पाँच सौ रुपये और पचास पैसे"
# input_text = "पाँच सौ रुपये छियानवे पैसे"
# input_text = "छियानवे पैसे"
# output = apply_fst(input_text, money.fst)
# print(output)
