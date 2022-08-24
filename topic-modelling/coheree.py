import cohere
from cohere.classify import Example
co = cohere.Client('ny1z17B9CMezuRnj0pMJo0PGqfsWVloqgpo3GaeR')
classifications = co.classify(
  model='medium',
  taskDescription='Classify these sentences as question, decision, task, or other',
  outputIndicator='Classify this sentence',
  inputs=["Rural customers need special help to feel more valued.", "Our sales teams need more accurate information on our customers.",
"A survey will be completed to collect data on spending habits in these areas.", 
"The results of this survey will be delivered to our sales teaMs.",
"We are considering specific data mining procedures to help deepen our understanding."],
  examples=[Example("I suggest we break up into groups and discuss the ideas we've seen presented.", "task"),
            Example("we need to give our rural sales teams better customer information reporting.", "task"), 
            Example("we need to return to our rural base by developing an advertising campaign to focus on their particular needs.", "task"),
            Example("go round the table first to get all of your input.", "task"),
            Example("just summarize the main points of the last meeting.", "task"),

            Example("We'll have to leave that to another time.", "decision"), 
            Example("Let's meet at the same time, 9 o'clock.", "decision"),
            Example("we have been focusing too much on urban customers and their needs.", "decision"),
            Example("rural customers want to feel as important as our customers living in cities.", "decision"),
            Example("I have to agree with Alice.", "decision"),

            Example("How does Friday in two weeks time sound to everyone", "question"),
            Example("Can we fix the next meeting, please", "question"), 
            Example("What exactly do you mean?", "question"), 
            Example("Is that OK for everyone?", "question"),
            Example("Have you all received a copy of today's agenda?", "question"),
            
            Example("Good idea Donald.", "other"),
            Example("I'd like to thank Jack for coming to our meeting today.", "other"),
            Example("As you can see, we are developing new methods to reach out to our rural customers.", "other"),
            Example("I must admit I never thought about rural sales that way before.", "other"),
            Example("Excuse me, I didn't catch that.", "other")
            ])

print('The confidence levels of the labels are: {}'.format(classifications.classifications))