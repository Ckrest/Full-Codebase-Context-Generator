from symspellpy import SymSpell


def create_symspell_from_terms(terms, max_distance=2, prefix_length=7):
    """Create a SymSpell dictionary from an iterable of terms."""
    sym = SymSpell(max_dictionary_edit_distance=max_distance, prefix_length=prefix_length)
    for term in terms:
        for token in str(term).split():
            sym.create_dictionary_entry(token, 1)
    return sym


def correct_phrase(symspell, phrase, max_edit_distance=2):
    """Return the best correction for ``phrase`` using ``symspell``."""
    if not symspell:
        return phrase
    suggestions = symspell.lookup_compound(phrase, max_edit_distance=max_edit_distance)
    return suggestions[0].term if suggestions else phrase
