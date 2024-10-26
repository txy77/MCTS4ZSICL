import os
import sys

ZS_ICL_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
sys.path.insert(0, ZS_ICL_ROOT_PATH)

from src.method.DAIL import DAIL
from src.method.DAIL_cali import DAIL_CALI
from src.method.SelfICL import SelfICL
from src.method.FS import FS
from src.method.ZS import ZS
from method.Search import Search

method2class = {
    "DAIL": DAIL,
    "DAIL_CALI": DAIL_CALI,
    "SelfICL": SelfICL,
    "FS": FS,
    "ZS": ZS,
    "Search": Search,
}