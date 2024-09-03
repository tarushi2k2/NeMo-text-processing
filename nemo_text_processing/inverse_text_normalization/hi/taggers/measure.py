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

# from nemo_text_processing.inverse_text_normalization.hi.taggers.decimal import DecimalFst
# from nemo_text_processing.inverse_text_normalization.hi.taggers.cardinal import CardinalFst
from pynini.lib import pynutil, rewrite

from nemo_text_processing.inverse_text_normalization.hi.graph_utils import (
    GraphFst,
    convert_space,
    delete_extra_space,
    delete_space,
    insert_space,
)
from nemo_text_processing.inverse_text_normalization.hi.utils import apply_fst, get_abs_path


class MeasureFst(GraphFst):
    """
    Finite state transducer for classifying measure
        e.g. ऋण बारह किलोग्राम -> measure { decimal { negative: "true"  integer_part: "12"  fractional_part: "50"} units: "kg" }

    Args:
        decimal: DecimalFst
        measure: MeasureFst
    """

    def __init__(self, decimal: GraphFst):
        super().__init__(name="measure", kind="classify")

        decimal_graph = decimal.final_graph_wo_negative
        # decimal_graph = decimal.final_graph

        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("ऋण", "\"true\"") + delete_extra_space, 0, 1,
        )

        weight_graph = pynini.string_file(get_abs_path("data/measure/weight.tsv")).invert()
        length_graph = pynini.string_file(get_abs_path("data/measure/length.tsv")).invert()
        area_graph = pynini.string_file(get_abs_path("data/measure/area.tsv")).invert()
        volume_graph = pynini.string_file(get_abs_path("data/measure/volume.tsv")).invert()
        temperature_graph = pynini.string_file(get_abs_path("data/measure/temperature.tsv")).invert()
        speed_graph = pynini.string_file(get_abs_path("data/measure/speed.tsv")).invert()

        self.weight = pynutil.insert("units: \"") + weight_graph + pynutil.insert("\" ")
        self.length = pynutil.insert("units: \"") + length_graph + pynutil.insert("\" ")
        self.area = pynutil.insert("units: \"") + area_graph + pynutil.insert("\" ")
        self.volume = pynutil.insert("units: \"") + volume_graph + pynutil.insert("\" ")
        self.temperature = pynutil.insert("units: \"") + temperature_graph + pynutil.insert("\" ")
        self.speed = pynutil.insert("units: \"") + speed_graph + pynutil.insert("\" ")

        graph_weight = (
            pynutil.insert("decimal { ")
            + optional_graph_negative
            + decimal_graph
            + pynutil.insert(" }")
            + delete_extra_space
            + self.weight
        )
        graph_length = (
            pynutil.insert("decimal { ")
            + optional_graph_negative
            + decimal_graph
            + pynutil.insert(" }")
            + delete_extra_space
            + self.length
        )
        graph_area = (
            pynutil.insert("decimal { ")
            + optional_graph_negative
            + decimal_graph
            + pynutil.insert(" }")
            + delete_extra_space
            + self.area
        )
        graph_volume = (
            pynutil.insert("decimal { ")
            + optional_graph_negative
            + decimal_graph
            + pynutil.insert(" }")
            + delete_extra_space
            + self.volume
        )
        graph_temperature = (
            pynutil.insert("decimal { ")
            + optional_graph_negative
            + decimal_graph
            + pynutil.insert(" }")
            + delete_extra_space
            + self.temperature
        )
        graph_speed = (
            pynutil.insert("decimal { ")
            + optional_graph_negative
            + decimal_graph
            + pynutil.insert(" }")
            + delete_extra_space
            + self.speed
        )

        graph = graph_weight | graph_length | graph_area | graph_volume | graph_temperature | graph_speed
        self.graph = graph.optimize()

        final_graph = self.add_tokens(graph)
        self.fst = final_graph


# cardinal = CardinalFst()
# decimal = DecimalFst(cardinal)
# measure = MeasureFst(decimal)
# input_text = "सत्रह दशमलव सात सात मील प्रति घंटा"
# input_text = "एक सौ एक दशमलव शून्य शून्य किलोमीटर प्रति घंटा"
# output = apply_fst(input_text, measure.fst)
# output = rewrite.top_rewrite(input_text, measure.fst)
# print(output)
# output = measure {decimal { negative = "true"  integer_part = "12"  fractional_part = "50"} unit: "kg" }
# tokens { decimal { integer_part: "१०१"  fractional_part: "६" } } tokens { name: "किलोग्राम" }
# १०१.६ किलोग्राम
