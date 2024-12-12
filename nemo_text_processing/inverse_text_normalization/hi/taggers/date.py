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
 
from nemo_text_processing.inverse_text_normalization.hi.graph_utils import (
    NEMO_HI_DIGIT,
    GraphFst,
    delete_extra_space,
    delete_space,
    insert_space,
)
from nemo_text_processing.inverse_text_normalization.hi.utils import get_abs_path, apply_fst
 
 
class DateFst(GraphFst):
    """
        Finite state transducer for classifying date, 
        e.g. पांच जनवरी दो हज़ार बारह -> date { month: "जनवरी" day: "५" year: "२०१२" preserve_order: true }
        e.g. दो हज़ार बारह -> date { year: "२०१२" preserve_order: true }     
    Args:
        cardinal: CardinalFst
        date: DateFst
    """
 
    def __init__(self, cardinal: GraphFst):
        super().__init__(name="date", kind="classify")
 
        graph_year = pynutil.add_weight(
            pynini.compose(cardinal.graph_no_exception, pynini.closure(NEMO_HI_DIGIT, 1, 4)), 0.03
        )
 
        month_graph = pynini.string_file(get_abs_path("data/date/months.tsv"))
        graph_date_days = pynini.string_file(get_abs_path("data/date/date_days.tsv")).invert()
        graph_century = pynini.string_file(get_abs_path("data/date/century.tsv")).invert()
 
 
        self.day = pynutil.insert("day: \"") + graph_date_days + pynutil.insert("\" ")
        self.month = pynutil.insert("month: \"") + month_graph + pynutil.insert("\" ")
        self.year = pynutil.insert("year: \"") + graph_year + pynutil.insert("\" ")
        self.century = pynutil.insert("text: \"") + graph_century + pynutil.insert("\" ")
        insert_comma = pynutil.insert(", ")
 
        graph_day_month = self.day + delete_space + self.month
        graph_month_day = self.month + delete_space + self.day
        graph_month_day += pynutil.insert(" preserve_order: true")
        graph_day_month_year = self.day + delete_space + self.month + delete_space + self.year
        graph_month_day_year = self.month + delete_space + self.day + delete_space + self.year
        graph_month_day_year += pynutil.insert(" preserve_order: true")
        graph_month_year = self.month + delete_space + self.year
        graph_saal = self.year
        graph_AD_BC = self.year + delete_space + self.century
        graph_day_month_year_century = self.day + delete_space + self.month + delete_space + self.year + delete_space + self.century
        graph_month_year_century = self.month + delete_space + self.year + delete_space + self.century
        graph_year_range = self.year + delete_space + pynutil.delete("से") + delete_space + self.year
 
 
        graph = graph_day_month | graph_month_day | graph_day_month_year | graph_month_day_year | graph_month_year | graph_saal | graph_AD_BC | graph_day_month_year_century | graph_month_year_century | graph_year_range
        final_graph = self.add_tokens(graph)
        self.fst = final_graph

#from nemo_text_processing.inverse_text_normalization.hi.taggers.cardinal import CardinalFst
#cardinal = CardinalFst()
#date = DateFst(cardinal)
#input_text = "पच्चीस मार्च दो हज़ार दस"
#input_text = "छ: मार्च उन्नीस सौ नब्बे"
#input_text = "छ: मार्च उन्नीस सौ नब्बे ईस्वी"
#input_text = "छह मार्च दो हज़ार दस"
#input_text = "तीन फ़रवरी"
#input_text = "चौवालीस सौ ईसा पूर्व"
#input_text = "फ़रवरी चौवालीस सौ ईसा पूर्व"
#input_text = "चौवालीस सौ ईस्वी"
#input_text = "उन्नीस सौ बीस से उन्नीस सौ छब्बीस"
#input_text = "उन्नीस सौ बीस से छब्बीस"
#output = apply_fst(input_text, date.fst)
#print(output)
