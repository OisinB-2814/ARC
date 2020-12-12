# The Abstraction and Reasoning Corpus Assignment (ARC) -  Oisin Brannock

The goal of this assignment is to pick 3 tasks from the training data and solve them manually. I found the task harder than I expected as there is so much choice and once you latch on to a problem its hard to let go and admit defeat. 

However in solving the three I chose I learned a lot about the data in general. In reading Chollet's paper *https://arxiv.org/pdf/1911.01547.pdf*, I found myself consistently thinking back to the following statement during my time with the problems: 
*'we need to be able to define and evaluate intelligence in a way that enables comparisons between two systems, as well as comparisons with humans.'* 
We put a lot of emphasis on humans vs. AI but I had never thought of it being incorrecly quantified until I read this paper. 

*'Intelligence lies in broad or general-purpose abilities; it is marked by flexibility and adaptability (i.e. skill-acquisition and generalization), rather than skill itself'* 

couldn't be a fairer statement. In designing my solutions I first thought how could I simply get all the correct training and test sequences done. However the further I delved the more I desired to generalize my functions to work with any compatible inputs; to allow for adaptabilty of my code into a possible machine learning process. I looked beyond the simple right solution and instead tried to think of my work in terms of Chollet's proposal for AI. I really feel I understand the goals of what we strive to achieve now more and I'm not just saying that for more marks I truly believe it :). 

In general all of these tasks can be solved with numpy functions acting upon arrays, along with basic python functionality like dictionaries, lambda functions and list slicing. The tasks really challenege you to think outside of your comfort zone and in doing so alongside Chollet's paper I feel I definitely achieved that and a new level of understanding to what I want to achieve. Not every task will have the same solution but I feel training a model to spot specific problem types and using specific functions to solve them would be the way to do it. 

Could all of these 400 tasks be solved simply using a model incorporating numpy? I certainly wouldn't have thought so before starting out but now I am more confident in that premise. On a final note my code falls into what I would understand to be the "Local generalization, or “robustness”" description used by Chollet in his paper. It can handle new data of the same kind of structure as before, but will fall down if something unforseen has come into play due to the hard code nature of the functions. Elevation to the next two stages is crucial to cut out any further human intervention.

