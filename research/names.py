# Candidate names for an object SAM may not ground by the name it was given: the name, its head noun, WordNet
# synonyms of the head, and nouns from the video's prompt (caption) that WordNet relates to the head (the same
# thing, a kind of it, or at most two steps more general). No labels are used: the caption is what the creator
# typed, and which candidate wins is decided by SAM's own scores (name_probe.py).
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

lemma = WordNetLemmatizer().lemmatize


def related(word, head):
    heads = set(wn.synsets(head, "n"))
    up = lambda s: s.hypernyms() + s.instance_hypernyms()
    for s in wn.synsets(word, "n"):
        if s in heads or heads & set(s.closure(up)):  # same, or word is a kind of head
            return True
        if any(s in set(h.closure(up, depth=2)) for h in heads):  # head is a kind of word, close by
            return True
    return False


def candidates(concept, caption=""):
    words = concept.lower().split()
    head, adj = words[-1], words[:-1]
    out = [concept.lower(), head]
    # Every sense: rare ones add junk ("testicle" for ball) but also the right one ("cube" for block, sense 5).
    # The junk is filtered later by requiring SAM to fire where the given name already weakly points.
    for s in wn.synsets(head, "n"):
        out += [n for n in (l.name().replace("_", " ").lower() for l in s.lemmas()) if n.replace(" ", "").isalpha()]
    tagged = pos_tag(word_tokenize(caption))
    for i, (w, t) in enumerate(tagged):
        n = lemma(w.lower())
        if t.startswith("NN") and n.isalpha() and related(n, head):
            out.append(n)
            if i and tagged[i - 1][1].startswith("JJ"):
                out.append(f"{tagged[i - 1][0].lower()} {n}")
    out += [" ".join(adj + [c]) for c in list(out) if adj and " " not in c]  # "yellow block" -> "yellow cube"
    return list(dict.fromkeys(out))


if __name__ == "__main__":
    c = candidates("animal", "A deer stands on a quiet road while a car passes by.")
    assert c[0] == "animal" and "deer" in c and "car" not in c and "road" not in c, c
    c = candidates("yellow block", "A child stacks a yellow cube on a wooden table.")
    assert "yellow block" in c and "block" in c and "table" not in c, c
    print("ok")
