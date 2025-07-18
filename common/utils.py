import random


_z_scores = {0.80: 1.28, 0.85: 1.44, 0.90: 1.64, 0.95: 1.96, 0.99: 2.58}


def print_prediction(predicted: str, expected: str):
    from termcolor import colored

    color = "green" if predicted == expected else "red"
    if color == "red":
        print(colored(f"Expected: {expected}", "blue"))

    print(colored(predicted, color))


def stratified_sampling(df, by_column, n_samples):
    frac = n_samples / len(df)
    strat_sample = df.groupby(by_column).sample(frac=frac, random_state=50)
    return strat_sample


def calculate_sample_size(
    population_size, confidence_level: float = 0.95, margin_error=0.05
):
    e2 = margin_error**2
    z2 = (_z_scores[confidence_level] ** 2) * 0.25
    z2_by_e2 = z2 / e2
    sample_size = z2_by_e2 / (1 + z2_by_e2 / population_size)
    return round(sample_size)


def get_sample(data: list):
    random.seed(50)
    return random.sample(data, calculate_sample_size(len(data)))


if __name__ == "__main__":
    assert calculate_sample_size(1000) == 278
