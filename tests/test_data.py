from src.data import (load_split, grammar_stats, check_sentence, agreement_probes, SPLITS,
                      default_tokenizer)


def test_splits_use_distinct_seeds_and_differ():
    seeds = [s for s, _ in SPLITS.values()]
    assert len(set(seeds)) == len(seeds)
    assert load_split("train")[:2000] != load_split("test")[:2000]


def test_source_text_is_fully_grammatical():
    st = grammar_stats(load_split("test")[:20000])
    assert st["frac_grammatical"] == 1.0 and st["frac_structure"] == 1.0


def test_checker_detects_errors():
    assert check_sentence("the dog near the cats runs")["agreement"] is True
    assert check_sentence("the dog near the cats run")["agreement"] is False
    assert check_sentence("two dog runs")["agreement"] is False
    assert check_sentence("the dog the runs")["structure"] is False
    assert check_sentence("the dgo runs")["words_valid"] is False


def test_probes_are_minimal_pairs():
    ps = agreement_probes(n=20)
    for p in ps:
        assert p["good"] != p["bad"]
        assert check_sentence((p["context"].split(". ")[-1] + p["good"])[:-1])["agreement"] is True
        assert check_sentence((p["context"].split(". ")[-1] + p["bad"])[:-1])["agreement"] is False


def test_tokenizer_roundtrip():
    tok = default_tokenizer()
    s = load_split("val")[:500]
    assert tok.decode(tok.encode(s)) == s
