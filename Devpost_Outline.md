# RachmaninoffNN: Style-Specific Music Generation with Biaxial LSTM

## Team Members
* Xin Lian (xlian1)
* Yanyu Tao (ytao5)
* Yezhi Pan (ypan34)
* Yongjeong Kim (ykim235)

## Introduction

Deep learning has shown its superiority in solving generative tasks in the domains of natural language processing and computer vision, creating artificial articles and pictures that are realistic and comparable to the works of humans. While deep learning has also been incorporated in the field of audio in recent years for automatic music generation, generating realistic and aesthetic music pieces remains challenging. Most existing neural network music generation algorithms specialize in creating new music in a particular music genre. Still, few algorithms possess a tunable ability that gives users the freedom to choose their desired musical style, including music genre, composer’s style, mood, etc. In the paper by Mao et al. [[1](#references)], the authors aim to create a model capable of composing music given a specific or a mixture of musical styles. They believe that such a model can help customize generated music for people in the music and film industries. They develop upon the previously introduced genre-agnostic algorithm, Biaxial LSTM, and incorporate new methods to learn music dynamics. In our project, we will be implementing the deep learning model presented in the paper, DeepJ, utilizing a different dataset with a mixed-style piano repertoire from the 17th to the early 20th century. 

## Related Work
One of the most notable related works that our target paper compares against is "Generating Polyphonic Music Using Tied Parallel Networks" by Daniel D. Johnson [[2](#references)]. This work is unique in the sense that it can compose music of multiple genres, whereas the previous works mainly focus on a music composition task of a specific genre. The author uses Bi-Axial LSTM and Tied-Parallel LSTM-NADE for the music generation and prediction tasks for the polyphonic music, where the models can take different melodies concurrently at the time-level granularity (joint probability distribution). As noted by both papers [[1](#references), [2](#references)], this related work achieved high prediction accuracy and generated the music of multiple genres but fails to ensure a strong consistency in the single-genre music (i.e., each output is of a particular genre). Our target paper aims to leverage this shortcoming. The link for the DeepJ code can be found in the reference section.

## Data

We will implement the model using the MAESTRO Dataset V3.0.0 [[3](#references)], distributed by Google’s Magenta program. Specifically, we will utilize the MIDI file portion of the data, which constitutes around 200 hours of recordings from the International Piano-e-Competition, 2004-2018. We will transcribe the MIDI files to piano roll representation of notes to align with the training data representation from the original paper, using the get_piano_roll() functionality from the pretty_midi library. We will also need to assign music styles (classical, baroque, or romantic) based on composers to each MIDI track, following the style assignment mechanism in the original manuscript.

## Methodology

Our implementation is primarily based on DeepJ's original implementation, where we use PyTorch and a different dataset instead. The original implementation has specific fixed codes that assume particular forms for the data (i.e., the dataset splits into each type of other genres) and constant values. We plan to build our logic for preprocess method that can convert this new dataset to work with our model. After the prepossessing, we will feed them into our model to generate the music, which requires human interpretation for the accuracy test. 

This model is Biaxial LSTMs for the time-based and node-based modules. This model architecture uses these LSTMs and conditional probabilities of concurrent melody at each timeframe to generate the unique music. It uses the convolution layer to extract the note features then these are passed into our two LSTMs with their appropriate contexts. The first LSTM is the time module, and the second one is the note module. We need to reimplement two types of hidden layers DeepJ [[1](#references)] uses: 1) one linear hidden layer is used to manipulate the style embedding, and 2) each LSTM works with another hidden layer with the tanh activation to create a non-linear version of the embedding. 

Following is a quick summary of the general model architecture explained in DeepJ [[1](#references)]. The model described in the paper starts with the convolution layer: a 1-dimensional convolution layer is applied to note input to extract note features. These note feature outputs are then concatenated with contextual inputs and incorporated information on style by adding a non-linear representation of style before being fed into the first LSTM layer (Time-Axis Module). The outputs from this LSTM layer also need to be concatenated with chosen notes from results, do component-wise sum with style, and then be fed into the next LSTM layer (Note-Axis Module). Finally, we provide the outputs of this layer into the sigmoid probability layer to get prediction results. Also, the key thing to note for this model’s weakness is that it relies on human input for the final evaluation. 

While reimplementing with a different framework sounds like a simple task, understanding the existing models without detailed code documentation can be challenging. Also, simply modifying the dataset and converting the Tensorflow-version to the PyTorch version does not necessarily make our model better. For instance, we will need to ensure that our implementation scales well with our GPU accelerator environment and the size of the dataset (e.g., we are given an option to choose the number of accelerators). Furthermore, creating a developing environment that works for every team member's local environment is crucial if we decide to implement the initial code locally for the testing and then deploy them to the cloud instance.

In summary, these are the current challenges we identified: 1) ensuring that we have a proper development environment that works for everyone (e.g., a container for the initial experiment or the appropriate cloud instance setting), 2) modifying the preprocessing logic to ensure compatibility with our dataset, 3) modifying the code to optimize the GPU accelerator fully (e.g., less branch-blocking and hyperparameters tuning given our rich GPU environment), and 4) scaling with the large dataset with careful management of coding styles. 

## Metrics

We plan to train our model on the MAESTRO dataset consisting of three specific styles: classical, baroque, and romantic. Following the original manuscript, we will also truncate the pitch range in the MIDI files to reduce the input dimensionality. Hyperparameters will be decided upon tuning, including the number of units for the two axes in the model, number of filters, and dimensions of the embedding space. The most optimal ones will be selected. The model will be updated using stochastic gradient descent with the Adam optimizer. Detailed training steps can be found in the [Methodology section](#methodology).   

Given that music evaluation can be personal, the authors launch subjective experiments to evaluate the generated music from two aspects, quality, and style. For quality analysis, general users are asked to choose a better one from pairs of music generation of DeepJ and music generation of the original Biaxial model. For style analysis, users with music backgrounds are divided into two groups. They are asked to make a manual classification of style given the music generations of DeepJ and authentic music pieces, respectively. A hypothesis test is conducted based on their responses to check the difference in identifiability of musical style between the two groups. We will also conduct a personal user study in evaluating the performance of our model. Still, due to the time limitation of the project, we may not find as many users as in the original paper, and we will calculate objective metrics such as perplexity score as supplementary to examine our model. 

In this project, our base goal is to successfully reimplement the DeepJ model and train the model on the MAESTRO dataset. We target to generate discernable, realistic music comparable to the work of humans, and our stretch goal is to create aesthetic music with identifiable styles to some extent. 

## Ethics

Since our project is not concerned with lyrics or any natural language-related elements in the music generation process, the primary ethical complication involved is intellectual property and copyright issues. Is one broader societal issue relevant to our chosen problem space: how should we distribute credits when using AI or DL models to engage in the creative process? To what extent is it ethical to make AL or DL models "learn" from other people's work? There are no agreements or legal terms that specify whether it's appropriate to generate music from copyrighted works, so we need to consider this from an ethical perspective. As a result, we will only train the model on properly licensed data and will not attempt to generate music from any web-scraped or unlicenced MIDI files.  

The significant stakeholders of music generation deep learning models are composers/songwriters of the music in the training dataset and the audience who listen to the music generated by the model. Some secondary stakeholders include music label companies in the industry and content creators who use the generated music. The quality of our model results is entirely subjective to the audience, so the stakeholders won't be affected by mistakes made by our model but by the deployment of our model if it ever gets commercialized. Future discussions should include topics such as compensation to individuals or companies who own the copyright of the training music for any profit made by the music generated from the model or potential decrease in creative labor cost and resulting in attrition in the creative industry.  

## Division of Labor

### Implementation

We understand that the implementation works cannot be evenly divided among members since it is difficult to predict the time/effort required for each project segment. Instead, we plan to create git branches for each member where we may hold weekly/bi-weekly meetings to synchronize the process and undergo code reviews. Then, we will merge with the main branch on an ongoing basis to ensure the quality of working codes. 

The attempted distribution of work is as follows:
- Data preprocessing (data cleaning, style assignment, compatibility modification, etc.) - Yezhi Pan
- Model architecture - Yongjeong Kim
- Test and main (calculate complexity, call model, etc.) - Yanyu Tao
- Visualization and other utility functions - Xin Lian
  

### Survey 
As noted from the target paper [[1](#references)], the essential metric of evaluating the DeepJ framework is surveying a diverse group of people, including but not limited to one with a Deep Learning background and another group without expertise.

The attempted distribution of work is as follows:
- Create survey templates and links - Yezhi Pan and Yanyu Tao
- Distribute surveys - all
- Analyze survey results and generate relevant charts - Yongjeong Kim and Xin Lian

### Final Report and Posting
Final report and posting preparations are crucial parts of the final project to demonstrate the bottleneck of our implementation through evaluations and description. We plan to hold weekly or daily meetings to ensure that everyone participates in summarizing and documenting the project and share each member's interpretation and understanding of the framework on their roles.

## References
[1] H. H. Mao, T. Shin and G. Cottrell, "DeepJ: Style-Specific Music Generation," *2018 IEEE 12th International Conference on Semantic Computing (ICSC), 2018*, pp. 377-382, doi: 10.1109/ICSC.2018.00077.  
* Paper [[link][target]]  
* Repo [[link][target_code]]  

[2] D. D. Johnson, “Generating polyphonic music using tied Parallel Networks,” *Computational Intelligence in Music, Sound, Art and Design*, pp. 128–143, 2017.
* Paper [[link][related_work_1]]  

[3] Magenta, "The MAESTRO Dataset," *magenta*. Updated October 29, 2018. [Website].  
* Dataset [[link][dataset]]  


[target]: https://ieeexplore.ieee.org/document/8334500
[target_code]: https://github.com/calclavia/DeepJ
[related_work_1]: https://link.springer.com/chapter/10.1007/978-3-319-55750-2_9
[dataset]: https://magenta.tensorflow.org/datasets/maestro#download
