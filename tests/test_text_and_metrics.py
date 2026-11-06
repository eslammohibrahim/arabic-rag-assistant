import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from metrics import contains, exact_match, f1, says_unknown  # noqa: E402
from text import light_stem, normalize, tokenize  # noqa: E402


def test_normalize_unifies_spelling():
    assert normalize("أَحْمَد") == normalize("احمد")
    assert normalize("مدرسة") == normalize("مدرسه")
    assert normalize("٢٠١٨") == "2018"


def test_light_stem_removes_article_and_prepositions():
    assert light_stem("المدينه") == "مدينه"
    assert light_stem("والمدينه") == "مدينه"
    assert light_stem("بالقاهره") == "قاهره"
    assert light_stem("ال") == "ال"  # too short to strip


def test_tokenize_drops_stopwords():
    assert tokenize("ما هي عاصمة اليابان؟") == ["عاصمه", "يابان"]


def test_exact_match_ignores_spelling_and_article():
    assert exact_match("القاهرة", "قاهره") == 1.0
    assert exact_match("الإسكندرية", "القاهرة") == 0.0


def test_f1_partial_overlap():
    assert f1("صحفي وإعلامي سعودي", "صحفي وإعلامي") == 0.8
    assert f1("", "شيء") == 0.0


def test_contains_and_unknown():
    assert contains("يبلغ طوله 6650 كيلومترا تقريبا", "6650 كيلومترا") == 1.0
    assert says_unknown("لا أعرف.")
    assert not says_unknown("القاهرة")
