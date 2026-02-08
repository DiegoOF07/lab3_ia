from parser import read_dataset
from spam_filter import SpamFilter
from training import train_test_split, print_results

def main():
    data = read_dataset("./data/dataset.txt")
    
    train_data, test_data = train_test_split(data, split=0.8)
    
    model = SpamFilter(k=1)
    model.train(train_data)
    
    print(f"Probabilidades a priori:")
    print(f"P(Spam) = {model.priors['spam']:.4f}")
    print(f"P(Ham) = {model.priors['ham']:.4f}")

    df_likelihoods = model.get_likelihoods()
    print(df_likelihoods.head())

    test_messages = [
        "Congratulations! You won a free prize. Click here now!",
        "Hello, how are you? See you tomorrow for lunch",
        "URGENT OFFER: Buy now and get 50% discount",
        "The report is ready for your review",
        "Win money easily from home without effort",
        "Thanks for your email, I'll reply soon"
    ]
    
    for msg in test_messages:
        prediction = model.predict(msg)
        print(f"Mensaje: {msg}")
        print(f"Predicción: {prediction.upper()}")

    results = model.evaluate(test_data)
    print_results(results)
    
if __name__ == "__main__":
    main()
