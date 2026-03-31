import pickle
from src.dataset import load_comtrans_data
from src.smt_model import StatisticalMT

print("Loading data...")
data = load_comtrans_data(num_sentences=3000) # Increased to 3000 for better quality since we are saving it ahead of time

print("Initializing models...")
model_en2de = StatisticalMT(direction="en2de")
model_de2en = StatisticalMT(direction="de2en")

print("Training EN->DE...")
model_en2de.train(data, iterations=5)

print("Training DE->EN...")
model_de2en.train(data, iterations=5)

print("Extracting serializable state...")
cache = {
    "en2de_trans": model_en2de.best_trans,
    "en2de_phrase": model_en2de.phrase_memory,
    "de2en_trans": model_de2en.best_trans,
    "de2en_phrase": model_de2en.phrase_memory,
}

print("Saving models to models.pkl...")
with open("models.pkl", "wb") as f:
    pickle.dump(cache, f)

print("Done pre-training!")
