# A Cognitive, Information-Theoretic Account of Online Misinformation

## Samuel Hutchinson, Senior Thesis Project in Cognitive Science, Advisor: Dr. Christopher Baldassano

![poster](/poster_1.png "Thesis Poster")
#Poster version that I presented at the Columbia Undergraduate Computer and Data Science Research Fair#

### Working drafts of the text for my senior thesis project on cross-entropy variation in misinformation

#### Introduction, Background, Relevance
“They’re lying to you,” announced the poster-board sign interrupting coverage from First Channel, a Kremlin-backed outlet and one of the main sources of news for Russian citizens. The protestor—who had briefly appeared onscreen to suggest that the Russian state media was producing disinformation regarding the war in Ukraine—was arrested within seconds (Krause-Jackson, 2022). A few weeks later, from April 6th to April 8th, 2022, The University of Chicago’s Institute of Politics and The Atlantic hosted a conference entitled “Disinformation and the Erosion of Democracy,” featuring prominent American politicians and legal scholars bemoaning the titular threat that they argue is spiraling into global catastrophe (Stern, 2022). With such high-profile examples in the media, the saliency and immediacy of mis- and disinformation spread online has quickly entered the international consciousness. 

In their particularly alarming study, Vosoughi et al. (2018) find that tweets containing false information are over 70 percent more likely to be retweeted—spreading “significantly farther, faster, deeper, and more broadly”—than tweets containing true information. As of 2018, the authors observed three main spikes in the total number of false tweets: the 2012 and 2016 American presidential elections and the 2014 Russian annexation of Crimea, suggesting that false political information is particularly salient. Leaked documents from Facebook show that of their own internal samples of English-language posts concerning COVID-19, two-thirds of the posts sampled contained “anti-vax” content (Schechner et al., 2021). What could explain this difference in virality between true and false information? What drives people to share misinformation at such inflated rates? Could answering this cognitive question point us in a potential direction to look for a solution to this problem? What form could this solution take—can we model the cognitive effects of misinformation in hopes of later identifying it when it occurs?

One place to start would be to consider the ways in which we already know misinformation-containing and non-misinformation-containing statements differ in their effects. Responses to false statements online indicate that these statements disproportionately inspire reactions of “surprise” or “shock” and “anger” or “disgust” in comparison to true statements’ inspiring of “sadness” and “trust” (Vosoughi et al., 2018; Pennycook & Rand, 2021; Kozyreva, et al., 2020; Quandt, 2018). This narrows the scope of our previous questions: from modeling misinformation, can we instead model statements that may inspire surprise or anger?

Recent advances in natural language processing (NLP) and work on large language models (LLMs) may help in answering this question. Sentiment classifiers—machine-learning algorithms designed to detect the emotional content of natural-language statements—are a well-established aspect of NLP research today. This opens one algorithmic door for this investigation: do sentiment classifiers rate statements containing misinformation as significantly more negative than others, thereby echoing the subjective responses of anger or disgust? What about the surprise element of misinformation? Goldstein et al. (2021) find, in ways which I will unpack later in this paper, that LLMs make similar predictions as humans in next-word prediction tasks. Of paramount importance to this investigation, however, they also find that LLMs and humans overlap in their confidence ratings and predictability judgements (calculated as cross-entropy, see Methods section below) regarding the words that do appear next in these tasks. These similarities suggest a possibility that LLMs could reliably identify statements that humans find surprising—thereby potentially providing the second piece to our algorithmic puzzle.

This brings me to the major questions of this investigation: can we use LLMs to model the differences in surprise and negative sentiment that humans subjectively attribute to misinformation? Can this model then make accurate classifications about whether or not unseen statements contain misinformation, as labeled by human raters? I hypothesize, based on the previous work highlighted above, that a classification model based on surprise and negative sentiment will indeed be able to accurately (at above-chance levels) distinguish between statements containing misinformation and those that do not. This result would not only indicate a short-term potential aid to online content moderators, but also point to a larger cognitive consequence—that what we find subjectively surprising in natural language statements can be accurately modeled as a prediction error, and this error can be reliably calculated by LLMs.

Such a result would prove both novel and significant for several reasons. As was also revealed in the leaked Facebook internal documents, human content moderators at large social media companies are struggling to stem the growing tide of misinformation on their platforms (Seetharaman et al., 2021). This highlights the need for alternative methods of flagging posts that potentially contain misinformation. Many studies (see Khan et al., 2021; Islam et al., 2020; Wang, 2017 for reviews) demonstrate that a variety of machine learning models perform well above chance at identifying statements that contain misinformation, but the models used are limited in explicability; every tested model is essentially a black box, lacking clear justifications for each classification (but see Shu et al., 2019). This investigation attempts to leverage the power and speed of algorithmic classification while maintaining explicability: instead of the traditional machine-learning strategy, the classifications will be done based on the pre-defined, cognitively-informed features of negative sentiment and surprise as measured by cross-entropy.

#### Methods
##### Sentiment
##### Cross-Entropy
##### Dataset
##### Models
###### Sentiment Classifier
###### GPT-2
###### Pre-trained
###### Fine-tuned
###### Combined Bayesian Classifier
#### Results
##### Cross-Entropy Distribution
![violin plot](/figures/fig2.png?raw=true "Distribution of CE by News Story Type and Model")
![density plot](/figures/fig2_1.png?raw=true "Distribution of CE by News Story Type")
##### Sentiment Distribution
![bar plot](/figures/fig4.png?raw=true "Distribution of Sentiment by News Story Type")
![density plot](/figures/fig4_1.png?raw=true "Distribution of Continuous Sentiment Score by News Story Type")
##### Joint Distribution
![2d plot](/figures/fig6.png?raw=true "2d Distribution of Sentiment and CE by News Story Type")

*Color-filled region represents the distribution of fake news stories, the empty region for true news stories*

![3d plot](/figures/fig6_1.png?raw=true "3d Distribution of Sentiment and CE by News Story Type")
##### Prediction/Classification
#### Discussion, Limitations, and Conclusion

#### References
Islam, M. R., Liu, S., Wang, Z., & Xu, G. (2020). Deep learning for misinformation detection on online social networks: a survey and new perspectives. Social Network Analysis and Mining, 10(82).

Khan, J. Y., Khondaker, T. I., Afroz, S., Uddin, G., & Iqbal, A. (2021). A benchmark study of machine learning models for online fake news detection. Machine Learning with Applications, 4.

Kozyreva, A., Lewandowsky, S., & Hertwig, R. (2020). Citizens versus the internet: confronting digital challenges with cognitive tools. Psychol. Sci. Public Interest, 21, 103-156.

Krause-Jackson, M. (2022, March 14). Putin’s State Media News Is Interrupted: ‘They’re Lying to You’. Bloomberg. https://www.bloomberg.com/news/articles/2022-03-14/putin-s-state-media-news-is-interrupted-they-re-lying-to-you

Pennycook, G. & Rand, D. (2021). The Psychology of Fake News. TiCS, 25(5), 388-402.

Quandt, T. (2018). Dark Participation. Media Commun., 6, 36-48.

Seetharaman, D., Horwitz, J., & Scheck, J. (2021, Oct. 17). Facebook Says AI Will Clean Up the Platform. Its Own Engineers Have Doubts. The Wall Street Journal. https://www.wsj.com/articles/facebook-ai-enforce-rules-engineers-doubtful-artificial-intelligence-11634338184?mod=article_inline

Schechner, S., Howritz, J., & Glazer, E. (2021, Sept. 17). How Facebook Hobbled Mark Zuckerberg’s Bid to Get America Vaccinated. The Wall Street Journal. https://www.wsj.com/articles/facebook-mark-zuckerberg-vaccinated-11631880296?mod=article_inline

Shu, K., Cui, L., Wang, S., Lee, D., & Liu, H. (2019). dEFEND: Explainable Fake News Detection. In The 25th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pp. 395-405.

Stern, J. (2022, April 7). Obama: I Underestimated the Threat of Disinformation. The Atlantic. https://www.theatlantic.com/ideas/archive/2022/04/barack-obama-interview-disinformation-ukraine/629496/

Vosoughi, S., Roy, D., & Aral, S. (2018). The spread of true and false news online. Science, 359(6380), 1146–1151. 

Wang, W. Y. (2017). “Liar, Liar Pants on Fire”: A New Benchmark Dataset for Fake News Detection. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pp. 422–426.
