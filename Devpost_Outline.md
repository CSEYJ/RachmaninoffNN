# RachmaninoffNN: Style-Specific Music Generation with Biaxial LSTM

## Team Members
* Xin Lian (xlian1)
* Yanyu Tao (ytao5)
* Yezhi Pan (ypan34)
* Yongjeong Kim (ykim235)

## Introduction

Deep learning has shown its superiority in solving generative tasks in the domains of natural language processing and computer vision, creating artificial articles and pictures that are realistic and comparable to the works of human. While deep learning has also been incorporated in the field of audio recent years for automatic music generation, generating realistic and aesthetic music pieces remains challenging. Most existing neural network music generation algorithms specialize to creating new music in a particular music genre, but few algorithms possess a tunable ability which gives users the freedom to choose their desired musical style including music genre, composer’s style, mood, etc. In the paper by Mao et al [[1](#references)], the authors aim to create a model that is capable of composing music given a specific or a mixture of musical styles, and they believe that such model can be helpful in customizing generated music for people in the music and film industries. They develop upon the previously introduced genre-agnostic algorithm, Biaxial LSTM, and incorporate new methods to learn music dynamics. In our project, we will be implementing the deep learning model introduced in the paper, DeepJ, utilizing a different dataset with mixed-style piano repertoire from the 17th to early 20th century.  

## Related Work
One of the most notable related works that our target paper compares against is "Generating Polyphonic Music Using Tied Parallel Networks" by Daniel D. Johnson [[2](#references)]. This work is unique in the sense that it can compose music of multiple genres, whereas the previous works mainly focus on a music composition task of a specific genre. The author uses Bi-Axial LSTM and Tied-Parallel LSTM-NADE for the music generation and prediction tasks for the polyphonic music, where the models can take different melodies concurrently at the time-level granularity (joint probability distribution). As noted by both papers [[1](#references), [2](#references)], this related work achieved high prediction accuracy and generated the music of multiple genres but fails to ensure a strong consistency in the single-genre music (i.e., each output is of a particular genre). Our target paper aims to leverage this shortcoming. The link for DeepJ code can be found at the reference section.

## Data

We will implement the model using the MAESTRO Dataset V3.0.0 [[3](#references)], distributed by Google’s Magenta program. Specifically, we are going to utilize the MIDI file portion of the data, which constitutes around 200 hours of recordings from International Piano-e-Competition, 2004-2018. We will transcribe the MIDI files to piano roll representation of notes, in order to align with the training data representation from the original paper, using the `get_piano_roll()` functionality from the `pretty_midi` library. We will also need to assign music styles (classical, baroque, or romantic) based on composers to each MIDI track, following the style assignment mechanism in the original manuscript.

## Methodology

To predict a note, model should have information of all previous time steps and all notes within the current time step that have already been generated. Hence, the Biaxial LSTM consists of two primary modules: the time-axis module and note-axis module. These two parts will take inputs along the time and note axes respectively.

The time-axis section consists of two-layer stacked LSTM units recurrent in time connected to each note octave (12 pitches above and below every note) in the input. The weights of each LSTM unit are shared across each note, forcing the time-axis section to learn note invariant features. The time axis section outputs features for each note and can be computed in parallel.

Taking in the note feature outputs from the time axis as input, the note-axis LSTM sweeps from the lowest note feature to the highest note feature to make predictions of each note conditioned on the predicted lower notes. The note-axis consists of another two-layer stacked LSTM that is recurrent in note. Each note’s features are first concatenated with the
lower chosen note. If the current note being considered is the lowest note, then zeros are concatenated.

Additionally, we use a linear hidden layer to linearly project the one-hot style input ***s*** to a style embedding ***h***. For each LSTM layer, we connect the style embedding ***h*** to another fully-connected hidden layer with tanh activation to produce a latent non-linear representation of style ***h’*** .

Then, let’s look at the total architecture. First, a 1-dimensional convolution layer is applied to note input to extract note features from each note's octave neighborhood. The outputs are then concatenated with contextual inputs and incorporated information of style through adding non-linear representation of style ***h’*** before fed into Time-Axis Module. The outputs from this LSTM layer also needs to be concatenated with chosen notes from results, do component-wise sum with ***h’*** and then be fed into next LSTM layer: Note-Axis Module. Finally, we feed the outputs of this layer into sigmoid probability layer to get prediction results.

## Metrics

We plan to train our model on the MAESTRO dataset consisting musics of three specific styles including classical, baroque, and romantic. Following the original manuscript, we will also truncate the pitch range in the MIDI files to reduce the input dimensionality. Hyperparameters including number of units for the two axes in the model, number of filters, and dimensions of the embedding space will be decided upon tuning, the most optimal ones will be selected. The model will be updated using stochastic gradient descent with the Adam optimizer. Detailed training steps can be found in the [Methodology section](#methodology). 

Given that music evaluation can be personal, the authors launch subjective experiments to evaluate the generated musics from two aspects, quality and style. For quality analysis, general users are asked to choose a better one from pairs of music generation of DeepJ and music generation of the original Biaxial model. For style analysis, users with music background are divided into two groups and are asked to make manual classification of style given the music generations of DeepJ and real music pieces respectively. A hypothesis test is conducted based on their responses to check the difference in identifiability of musical style between the two groups. We will also conduct subjective user study in evaluating the performance of our model, but due to the time limitation of the project, we may not find as many users as in the original paper, and we will calculate objective metrics such as perplexity score as supplementary to examine our model. 

In this project, our base goal is to successfully reimplement the DeepJ model and train the model on the MAESTRO dataset. We target at generating discernable, realistic music that are comparable to the work of human, and our strech goal is to generate aesthetic music with identifiable styles to some extent. 

## Ethics

Since our project is not concerned with lyrics or any natural language related elements in the music generation process, the major ethical complication involved is related to intellectual property and copyright issues. One broader societal issue relevant to our chosen problem space is: how should we distribute credits when using AI or DL models to engage in creative process? To what extent is it ethical to make AL or DL models to "learn" from other people's work? There is no agreements or legal terms that specify whether it's appropriate to generate music from copyrighted works, so we need to consider this from ethical perspectives. As a result, we will only train the model on properly licenced data and will not attempt to generate music from any web-scraped or unlicenced MIDI files.

The major stakeholders of music generation deep learning models are composers/songwriters of the music in the training dataset and the audience who listen to the music generated by the meodel. Some secondary stakeholders include music label companies in the industry and content creators who use the generated music. The quality of our model results is entirely subjective to the audience, so the stakeholders won't be affected by mistakes made by our model, but by the deployment of our model if it ever gets commercialized. Future discussions should include topics such as compensation to individuals or companies who own the copyright of the training music for any profit made by the music generated from the model, or potential decrease in creative labor cost and resulted attrition in the creative industry.

## Division of Labor

### Implementation
We understand that the implementation works cannot be evenly divided among members since it is difficult to predict the time/effort required for each project segment. Instead, we plan to create git branches for each member where we may hold weekly/bi-weekly meetings to synchronize the process and undergo code reviews. Then, we will merge with the main branch on ongoing basis to ensure the quality of working codes. We do not clearly distinguish code division, but we are currently considering each person working on the different python scripts (e.g., train script, preprocess script, etc.).

### Survey
As noted from the target paper [[1](#references)], the essential metric of evaluating the DeepJ framework is surveying a diverse group of people, including but not limited to one with a Deep Learning background and another group without expertise. While we do not have a clear division of the work for the survey process, the workloads include creating the relevant form and links to the generated music, sending a survey to different groups of people, and generating relevant charts to evaluate our results. We plan to set out the specific roles ongoing basis.

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
