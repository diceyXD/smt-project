from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def evaluate_translation(reference, hypothesis):
    """
    Calculates BLEU score for a single translation.
    reference: string
    hypothesis: string
    """
    # Reference expects a list of list of tokens for multiple references.
    # We provide a single reference.
    ref_tokens = [reference.lower().split()]
    hyp_tokens = hypothesis.lower().split()
    
    # Use smoothing function to handle short sentences / 0 n-gram counts
    cc = SmoothingFunction()
    
    # Calculate BLEU score
    score = sentence_bleu(ref_tokens, hyp_tokens, smoothing_function=cc.method1)
    return score
