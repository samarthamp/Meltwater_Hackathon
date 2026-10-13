# Overview

Build a model that plays Hangman on the provided word list — guessing letters one at a time, with at most 6 wrong guesses per word, to reveal as many hidden words as possible from a secret test set you'll never see during the contest.

This is a code-based competition: you submit a Notebook, not a predictions file. Scoring is done by running your Notebook's submitted code against the test set directly.

# Description

Classic Hangman. A secret word is hidden as underscores, and you guess one letter at a time:

- Correct letter → all occurrences revealed.
- Wrong letter → counts against your 6 allowed wrong guesses.
- Win: word fully revealed before 6 wrong guesses.
- Loss: 6th wrong guess reached with letters still hidden.

No non-alphabet characters will be present in the test set. All words are lowercase, and your guesses must also be lowercase letters.

# Evaluation

Your notebook must generate a standard flat CSV file containing exactly 250,000 rows. Instead of submitting an interactive function, your algorithm must pre-process all test words and output your complete, chronological sequence of intended guesses per word.

The file must contain a header row and follow this exact schema:

- `word_id`: The unique identifier matching the test set (0 to 249,999).
- `guessed_letters_string`: A single string of your lowercase letter guesses from left to right (e.g., etaonsrhld). Do not include spaces, commas, or brackets.

Note: Training outside Kaggle is allowed if you find the platform's runtime insufficient. However, your results must be fully reproducible by training on the provided train dataset using your submitted approach — submissions that cannot be replicated this way will not be considered. Note that API calls to any external model are not permitted and will fail during Kaggle submission evaluation. 

Score. Let N be the total number of words in the secret test set (N = 250,000). For each word i, let w_i (either 0 or 1) indicate whether it was solved (fully revealed within 6 wrong guesses). The primary metric is the win rate percentage:

$$
\text{Score} = \frac{\sum_{i=0}^{N-1} w_i}{N} \times 100
$$


Tie-break. Let E be the total number of wrong guesses made across all words up to the moment each game naturally terminates (either via a win or striking out at 6 lives). Among submissions with an equal primary score, rank is determined by a fine-grained efficiency penalty subtracted behind the decimal point:

$$
\text{LeaderboardScore} = \text{Score} - \frac{E}{100000000}
$$

A submission that solves more words always outranks one that solves fewer, regardless of efficiency—the tie-break fraction only decides ties on the core win rate itself.

# Constraints & Edge Cases

- The 6-Strike Lockout: The simulation loop for each word terminates instantly the moment your  `guessed_letters_string` triggers a 6th wrong guess or completes the word. Any remaining trailing letters in your string for that word are locked out and ignored.
- What Counts as a Strike?: Any character in your string that does not actively reveal a new letter in the hidden word—including an incorrect letter, a repeated/duplicate letter, a space, a number, or an invalid symbol—counts as exactly 1 wrong guess.


# Implementation Blueprint

## How to Build Your Model & Submission Loop

To maximize model flexibility and keep background evaluations ultra-fast, you must implement your own interactive game loop inside your notebook to generate your final submission strings.

Your algorithm cannot just output a static guess list; it needs to simulate the game locally, adapting turn-by-turn based on simulated hits and misses using the provided public test words.

## Step-by-Step Implementation Blueprint:

1. Read the Test Words: Open the public test.txt file and iterate through the words line by line.
2. Initialize the Game State: For each word, start with an array of underscores matching its exact character length (e.g., _ _ _ _ _).
3. Predict the Next Letter: Feed the current board mask (and your list of already guessed letters) into your model (Transformer, LSTM, etc.) to predict the highest-probability next letter.
4. Simulate the Reveal: Check your model's predicted letter against the test word:
    - If correct, reveal all instances of that letter instantly (e.g., guessing p changes _ _ _ _ _ to _ p p _ _).
    - If incorrect, log a wrong guess strike.
5. Loop Until Termination: Repeat steps 3 and 4 sequentially using the updated board masks until you either completely solve the word or accumulate 6 wrong guesses.
6. Record the Sequence: Join all the letters you guessed during that specific game run into a single flat string (e.g., eaoitns).

Export this final sequence into a file named exactly submission.csv containing the explicit line index as your tracking word_id.

## Example Submission Generation Code Structure:

```python
import pandas as pd

# Load the public test words list
with open("Data/test.txt", "r") as f:
    test_words = [line.strip().lower() for line in f if line.strip()]

rows = []

# Enumerate ensures the word_id matches the dataset line order perfectly
for word_id, test_word in enumerate(test_words):
    board = ["_"] * len(test_word)
    guessed_letters = []
    wrong_guesses = 0

    while wrong_guesses <= 6 and "_" in board:
        # Pass the updated board mask and history to your custom model
        next_guess = my_model.predict("".join(board), guessed_letters)
        guessed_letters.append(next_guess)

        if next_guess in test_word:
            # Update all matching letter positions instantly
            board = [test_word[i] if test_word[i] == next_guess else board[i] for i in range(len(test_word))]
        else:
            wrong_guesses += 1

    rows.append({
        "word_id": word_id,
        "guessed_letters_string": "".join(guessed_letters)
    })

# Save the final structured output file
pd.DataFrame(rows).to_csv("submission.csv", index=False)

```
