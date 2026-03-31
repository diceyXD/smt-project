import nltk
import ssl

def load_comtrans_data(num_sentences=5000):
    """
    Downloads and loads the EN-DE aligned sentences from NLTK's comtrans corpus.
    Limits to `num_sentences` to ensure manageable training time.
    """
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download('comtrans', quiet=True)
    from nltk.corpus import comtrans
    print("Loading comtrans DE-EN corpus...")
    
    # Load aligned sentences
    raw_sents = comtrans.aligned_sents('alignment-de-en.txt')
    
    if num_sentences:
        raw_sents = raw_sents[:num_sentences]
        
    return raw_sents

if __name__ == "__main__":
    sents = load_comtrans_data(2)
    for i, sent in enumerate(sents):
        print(f"Sentence {i+1}:")
        print(f"Words: {' '.join(sent.words)}")
        print(f"Mots: {' '.join(sent.mots)}")
        print(f"Alignment: {sent.alignment}")
        print("-" * 20)
