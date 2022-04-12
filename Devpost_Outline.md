# Devpost Outline

## Project Name
RachmaninoffNN

## Team Members
* Xin Lian (xlian1)
* Yanyu Tao (ytao5)
* Yezhi Pan (ypan34)
* Yongjeong Kim (ykim235)

## Introduction

## Related Work
One of the most notable related works that our target paper compares against is "Generating Polyphonic Music Using Tied Parallel Networks" by Daniel D. Johnson [[2][related_work_1]]. This work is unique in the sense that it can compose music of multiple genres, whereas the previous works mainly focus on a music composition task of a specific genre. The author uses Bi-Axial LSTM and Tied-Parallel LSTM-NADE for the music generation and prediction tasks for the polyphonic music, where the models can take different melodies concurrently at the time-level granularity (joint probability distribution). As noted by both papers [[1][target], [2][related_work_1]], this related work achieved high prediction accuracy and generated the music of multiple genres but fails to ensure a strong consistency in the single-genre music (i.e., each output is of a particular genre). Our target paper aims to leverage this shortcoming. The link for DeepJ codes can be found at the reference section.

## Data

## Methodology

## Metrics

## Ethics

## Division of Labor

### Implementation
We understand that the implementation works cannot be evenly divided among members since it is difficult to predict the time/effort required for each project segment. Instead, we plan to create git branches for each member where we may hold weekly/bi-weekly meetings to synchronize the process and undergo code reviews. Then, we will merge with the main branch on ongoing basis to ensure the quality of working codes. We do not clearly distinguish code division, but we are currently considering each person working on the different python scripts (e.g., train script, preprocess script, etc.).

### Survey
As noted from the target paper [[1][target]], the essential metric of evaluating the DeepJ framework is surveying a diverse group of people, including but not limited to one with a Deep Learning background and another group without expertise. While we do not have a clear division of the work for the survey process, the workloads include creating the relevant form and links to the generated music, sending a survey to different groups of people, and generating relevant charts to evaluate our results. We plan to set out the specific roles ongoing basis.

### Final Report and Posting
Final report and posting preparations are crucial parts of the final project to demonstrate the bottleneck of our implementation through evaluations and description. We plan to hold weekly or daily meetings to ensure that everyone participates in summarizing and documenting the project and share each member's interpretation and understanding of the framework on their roles.

### References
[1] H. H. Mao, T. Shin and G. Cottrell, "DeepJ: Style-Specific Music Generation," 2018 IEEE 12th International Conference on Semantic Computing (ICSC), 2018, pp. 377-382, doi: 10.1109/ICSC.2018.00077.  
* Paper [[link][target]]  
* Repo [[link][target_code]]  

[2] J. D.D., “Generating polyphonic music using tied parallel networks,” Correia J., Ciesielski V., Liapis A. (eds) Computational Intelligence in Music, Sound, Art and Design. EvoMUSART 2017., 2017. [Online]. Available: https://link.springer.com/chapter/10.1007/978-3-319-55750-2_9 978-3-319-55750-2 9  
* Paper [[link][related_work_1]]  



[target]: https://ieeexplore.ieee.org/document/8334500
[target_code]: https://github.com/calclavia/DeepJ
[related_work_1]: https://link.springer.com/chapter/10.1007/978-3-319-55750-2_9
