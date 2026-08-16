## NLP Ticket Categorizer

NLP - Natural Language Processing 
This project uses the NLP and Multinomial Naive Bayes to automatically categorize support tickets and route them to appropriate team.  

#### Training Data
 This project includes small labeled data for demonstration.
 The dataset is hard-coded inside `load_dummy_data()` function in `main.py`

## How it Works
 This Project uses the Multinomial Naive Bayes from scikit-learn to categorize input text.

### How the input ticket get categorize

##### Text Preprocessing
 The input undergoes cleaning to remove,
   - special character (e.g., !@#, punctuation).
   - Stopwords (common words like "the," "is," "and"). The text is then normalized to lowercase for consistency."
 Converts the input into lowercase

##### Feature Extraction (TF-IDF Vectorization)
 The cleaned text is transformed into numerical vectors using TF-IDF (Term Frequency-Inverse Document Frequency).

##### Classification using Naive Bayes
 The TF-IDF vectors are passed to `MultinomialNB` Model, which predicts the ticket's category based on learned patterns.


## How to run
#### Prerequisites
- Python 3.10 or greater
- [https://docs.astral.sh/uv/getting-started/](uv) - A fast Python package manager written in Rust.

#### Steps to run
- Step 1: Clone the repo
` git clone https://github.com/Byte-256/NLP_TICKET_CATEGORIZER `

- Step 2: Change directory
` cd NLP_TICKET_CATEGORIZER `

- Step 3: Install dependencies
` uv sync `

- Step 4: Run the Script
` uv run main.py `

 ## Project File Structure
 ```NLP_TICKET_CATEGORIZER/
├── main.py               # Main script
├── pyproject.toml        # Project dependencies
└── README.md             # This file
```
 
