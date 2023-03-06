# Classifying Online Misinformation with Cognitively-Informed Features from Transformer Language Models

## Samuel Hutchinson, Senior Thesis Project in Cognitive Science, Advisor: Dr. Christopher Baldassano

![poster](/poster_1.png "Thesis Poster")
*Poster version that I presented at the Columbia Undergraduate Computer and Data Science Research Fair*

### Paper Abstract

Online misinformation has crept into the public consciousness through discourse on topics ranging from presidential elections to the COVID-19 pandemic. Misinformation-containing statements spread both farther and faster than true statements on social networks, suggesting the need for an interpretable, algorithmic flagging mechanism. This investigation endeavors to devise such an algorithm based on cognitively-plausible explanations for misinformation’s virality: informational novelty and emotional valence. I use next-word prediction error signals from a GPT-2 model fine-tuned on true news stories from Reuters to assess the novelty component of this problem and a pre-trained RoBERTa-based sentiment classifier for the emotional-valence component. To create a classification model, I calculated the joint distribution of these errors and sentiments over a subset of true news stories and misinformation-containing stories from the ISOT Fake News dataset. I then used this joint distribution to predict the likelihood that unseen news stories contain misinformation. This model classifies news stories with around 79.4% accuracy, well above chance levels, furthering prior work showing similarity in next-word prediction between human readers and generative models like GPT-2. These results also indicate that online misinformation may be classifiable through computable and cognitively-interpretable natural-language metrics.

### Link to Working Draft of Paper
https://docs.google.com/document/d/1tO-2TbkqmrY5R5wKLBQ1-amV3wZwK8BZeATwepY37X8/edit?usp=sharing
