"Your ML model is only useful if it is helping you make better decisions". 
https://www.canva.com/design/DAFwmkqdrPI/KfAA-Lz9e1148wHcaxr-tQ/edit

"Predictive optimization", and this is truly amazing, a checklist of things to look for! Connect this to a quote in seeing like a state (it is a thin simplification?). Even if the philosphical questions of "legitimacy" seem abstract to you, there are important practical consequence if your prediction and decision models are not well-aligned

There is a strange disconnect I notice over and over in my work, as long as I've been in the Data Science-ing business. The disconnect is clearest to me when I compare the way I talk to other Data Scientists as opposed to the stakeholders that we partner with. 

When I talk to other Data Scientists, I speak in the formalisms I learned in my education about ML modeling, plus the technical material I've read since. In this conversation, the main thing we discuss is the predictive model, which takes in some inputs $X$ and produces a prediction $y$, usually an expected value or a probability. Our shared picture of the world usually looks like this:

[Image]

When I talk with Data Scientists, we spend a lot of time discussing the details of this prediction machine. We discuss its inputs, its outputs, the statistical properties of the algorithm we used to generate it, its ability to create predictions on out-of-sample data points, etc. If I'm building a system that detects tumors in X-ray images, we'll spend a lot of time thinking about how to get a well-calibrated model for $mathbb{P}$(Tumor | X-ray). We'll usually include a cross-validation procedure and a metric of prediction quality like the ROC-AUC. Basically, we're talking about everything you might find in a book like ESL. So far, so good - this is all familiar material, and there are plenty of useful packages on PyPI to help me do it.

When I talk to stakeholders who aren't Data Scientists, though, this prediction machine is usually not the thing we focus on. Instead, I often find that they're thinking about how to use the predictions to make the best choice. People in this position aren't asking "How do we calculate the probability that a tumor is present?". Instead, they're asking "How do I use the X-ray to decide who should get further testing or treatment?". These folks are looking not for predictions, but for decisions. Their key process is similar but not quite the same as the predictive picture:

[Image]

A stakeholder ultimately sees the prediction as a means to the larger end of decision-making. If you provide them a way to make a prediction, well, great! But there is still a gap to be addressed - the gap between receiving the prediction from the model and actually doing something with it.

When I first started working as a Data Scientist, this gap tripped me up all the time. I had spent a lot of time studying books like ESL, after all, and so I assumed that's what the job was. I'd frequently make elaborate presentations with dense, complex diagrams explaining my clever modeling choices and demonstrating my fantastic cross-validation scores. Some poor Product Manager would have to hear me blather on for thirty minutes about feature selection and the ROC curve, and then politely respond with "okay, of course that is all very interesting, but how does this actually help me do something?". This was frustrating for both of us. How should their decision process relate to the prediction model? 

As I mentioned before, the prediction model is a means to an end. It helps us make some decision better than we would have without it. That means that the predictive model is called by the decision process; it takes the data and provides some useful advice which we use to make the decision . For our X-Ray system, having the analysis of the image lets us decide which patients to test further; this decision is the ✨raison d'être✨ of the prediction. So the whole system looks something like this:

[Image]

This isn't just a small semantic difference. We need to think carefully about the decision system that our ML model is embedded in. Failure to do so leads to models with good CV scores which are unable to actually add any value in production.

Possibilities and Pitfalls in Prediction-Decision alignment

The predictive optimization idea; 

Algorithm 0 - the easieast decision + prediction framework is acting if some probability is greater than a threshold

Algorithm 1 - the easiest decision + prediction framework is maximizing expected value. 

Algorithm 2 - may also consider maximizing the "worst case"

?
What kinds of pitfalls show up here?s

Model vs Decision evaluation

Model metrics are not decision metrics

You should try and define a decision metric

A special case of this is the bender post

Ideally, you run an experiment (or if you can't do that, something quasi-experimental)

Compare with baseline somehow

Appendix: A crash course in defining your Decision process

Define the flowchart

Define the decision quality metric

Think about the pitfalls
