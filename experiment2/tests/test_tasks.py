import random

from exp2.tasks import (ALL_VARIANTS, VOCAB, agreement_probes, check_sentence, encode, grammar_score, instance,
                        split_text)


def test_all_text_in_fixed_vocab():
    for t, v in ALL_VARIANTS:
        assert set(split_text(t, v, "train", 3000)) <= set(VOCAB)


def test_train_and_test_instances_disjoint():
    for t, v in ALL_VARIANTS:
        if t == "grammar":
            continue
        rng = random.Random(1)
        tr = {instance(rng, t, v, "train")[0] for _ in range(2000)}
        te = {instance(rng, t, v, "test")[0] for _ in range(500)}
        assert not tr & te, (t, v)


def test_answers_correct():
    rng = random.Random(0)
    p, a = instance(rng, "symbol", "rev", "test")
    assert a[:-1] == p[:-1][::-1]
    p, a = instance(rng, "arith", "add", "test")
    x, y = p[:-1].split("+")
    assert int(a[:-1]) == int(x) + int(y)


def test_grammar_checker_and_variants():
    assert check_sentence("the dog near the cats runs", "svo")
    assert not check_sentence("the dog near the cats run", "svo")
    assert check_sentence("the dog the cats sees", "sov")
    assert not check_sentence("the dog sees the cats", "sov")
    assert check_sentence("the dog sees the cats", "svo")
    assert grammar_score(split_text("grammar", "sov", "test", 4000), "sov")[0] == 1.0
    for ctx, g, b in agreement_probes("sov", n=10):
        assert g != b


def test_encode_fixed_vocab_size():
    assert len(VOCAB) == 42 and encode("ab") == [VOCAB.index("a"), VOCAB.index("b")]
