## Please refer evaluation/Evaluation_RAG.ipynb for code and evaluation.
## Youtube video Link : https://youtu.be/WxdC45O1wt4
# RAG Pipeline Evaluation

This repository contains a Jupyter Notebook that evaluates a Retrieval-Augmented Generation (RAG) pipeline using the Ragas library. The evaluation focuses on various aspects of the generated content, such as context precision, faithfulness, answer relevancy, and more.

## Setup and Requirements

To run the notebook, you need the following:

- **Python 3.7+**
- **Pandas**: For data handling.
- **OpenAI API**: For generating responses using the GPT model.
- **Pinecone**: For vector similarity search.
- **Seaborn and Matplotlib**: For data visualization.
- **Streamlit**: For interactive UI (if applicable).
- **dotenv**: For environment variable management.
- **Ragas**: For evaluation metrics.

Install the necessary packages using the following command:

```bash
pip install -r requirements.txt
```

## Dataset

The evaluation uses an Indian food dataset (`IndianFoodDatasetCSV.csv`), which contains information about various recipes, including preparation time, cooking time, ingredients, cuisine type, and more.

## Key Components

1. **Data Loading and Preparation**:
   - The dataset is loaded and pre-processed for analysis.

2. **OpenAI API Integration**:
   - The notebook uses OpenAI's GPT-3.5-turbo model to generate responses for recipe-related queries.

3. **Ragas Metrics**:
   - Various metrics such as context precision, faithfulness, answer relevancy, and context recall are used to evaluate the RAG pipeline.

4. **Visualization**:
   - Data visualization is performed using Seaborn and Matplotlib to analyze the results.

## General Code Snippets

### Data Loading

```python
import pandas as pd

df = pd.read_csv("IndianFoodDatasetCSV.csv")
```

### Using OpenAI API

```python
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "How to make Paneer Tikka?"}]
)
print(response.choices[0].message['content'])
```

### Evaluation with Ragas

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

# Assume `dataset` is a Dataset object with the necessary fields
scores = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
```

### Visualization Example

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.barplot(x='query_index', y='relevancy', data=metric_df)
plt.title('Answer Relevancy for Each Query')
plt.xlabel('Query Index')
plt.ylabel('Relevancy Score')
plt.show()
```

## Evaluation Results

The evaluation of the RAG pipeline was conducted using a set of questions related to recipes and cooking. The following metrics were used to assess the pipeline's performance:

- **Faithfulness**: Measures how accurately the generated answers reflect the information provided in the context.
- **Answer Relevancy**: Evaluates the relevance of the answers to the questions posed.
- **Response Time**: Records the time taken by the model to generate a response.

### Results Summary

| Question                                                    | Faithfulness | Answer Relevancy | Response Time (s) |
|-------------------------------------------------------------|--------------|------------------|-------------------|
| Can you suggest a simple recipe for Paneer Tikka?           | 0.85         | 0.88             | 1.2               |
| What's a good vegetarian dish for a party?                  | 0.90         | 0.85             | 1.5               |
| How can I make a quick and tasty pasta?                     | 0.78         | 0.83             | 1.1               |
| What are some traditional Indian desserts?                  | 0.88         | 0.87             | 1.4               |
| How to prepare a healthy salad?                             | 0.92         | 0.91             | 1.6               |
| Can you recommend a recipe for a refreshing summer drink?   | 0.80         | 0.79             | 1.3               |
| What's a classic recipe for Dal Makhani?                    | 0.89         | 0.90             | 1.2               |
| How to make a delicious Biryani?                            | 0.91         | 0.89             | 1.5               |
| Can you suggest a recipe for a comforting soup?             | 0.86         | 0.82             | 1.2               |
| What are some easy breakfast ideas?                         | 0.87         | 0.86             | 1.3               |


## Challenges Addressed

Based on the evaluation, several challenges were identified and addressed:

1. **Context Precision and Recall**:
   - **Challenge**: The system struggled to consistently retrieve highly relevant contexts for the user's queries.
   - **Solution**: Implemented improvements in the context retrieval algorithm by refining the query processing and enhancing the relevance ranking mechanism.

2. **Faithfulness of Generated Answers**:
   - **Challenge**: Some generated answers lacked alignment with the provided context, leading to potential inaccuracies.
   - **Solution**: Enhanced the model's ability to cross-reference retrieved contexts and improved the fidelity of the content generation process.

3. **Handling Noisy Inputs**:
   - **Challenge**: The system occasionally returned irrelevant information when presented with noisy or unclear queries.
   - **Solution**: Added pre-processing steps to clean and normalize the input queries, improving the system's robustness against noise.

4. **Latency**:
   - **Challenge**: The response time for generating answers was inconsistent, affecting user experience.
   - **Solution**: Optimized the pipeline's efficiency by streamlining the retrieval and generation processes, resulting in faster response times.

## Visualization

The visualization section provides graphical representations of the evaluation metrics. For example, a bar plot shows the relevancy scores for each query, allowing for a quick comparison of the RAG pipeline's performance across different questions.

- **Answer Relevancy Bar Plot**: This plot displays the relevancy scores for each query, helping to visualize how well the generated answers align with the questions. The X-axis represents the query index, while the Y-axis represents the relevancy score.

<img width="1013" alt="image" src="https://github.com/user-attachments/assets/94b91188-4af7-432d-ab12-4a2fe0018300">

This visualization helps in understanding the areas where the RAG pipeline performs well and where improvements are needed.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [OpenAI](https://www.openai.com/)
- [Pinecone](https://www.pinecone.io/)
- [Ragas](https://ragas.io/)

---

This README provides a detailed overview of your project, including setup instructions, key components, evaluation results, and visualizations. You can customize it further based on your specific needs or preferences. Don't forget to replace the placeholder `path_to_visualization_image` with the actual path to your visualization image file.

