# How-to-Talk-of-Prediction-Problems

December 21, 2020

I appreciate comments. Shoot me an email at noel_s_cruz@yahoo.com!

Hire me! 😊

- So today, we will get a feel for machine learning
by looking at one of the oldest
and most enduring methods of classification:
Nearest Neighbor.
This will let us understand what
a classification problem is and also
introduce some basic machine learning terminology
like training set, test set,
training error, and test error.
We'll see how data can be represented as vectors
and Euclidean space and how we can compute
distances in spaces like this.
All this will give us the background we need
to put the nearest neighbor classifier into action.
So, the specific problem we will solve today
is the following.
We're given a picture or image of a handwritten digit
and we want to say what digit it is.
This, for example, is a three.
Here are some more examples: we have a zero,
a one, a two, and so on and so forth.
So how might we approach this problem?
Here's an idea.
A zero has got one loop in it.
An eight has got two loops.
A one has a single straight line.
A four has three straight lines, and so on.
What if we had a piece of software that took an image
and figured out how many loops it had
and how many straight lines, and also the relative positions
of these loops and straight lines.
Maybe we could then use this information
and write down a bunch of simple rules
to decide what digit it is.
Actually, people tried this a long time ago
and they ran into a lot of problems.
So first of all, handwritten digits are super noisy
and so it's hard to robustly pull out this information
about the loops and lines and so on.
Then, there's a lot of variability
in the way people write fours, sevens, nines, et cetera.
What this meant was that a huge number of rules were needed
to account for all the different special cases
and then, after all this trouble,
the systems didn't really work well at all.
So what we'll look at today is
an entirely different approach,
the machine learning approach.
Rather than trying to figure out
the underlying patterns ourselves,
we'll just let the machine do it for us.
Now, in order to figure out the patterns,
what the machine needs above all else
is a huge amount of data.
So what we do, is we assemble a large data set
of handwritten digit images,
each labeled with its correct image, with the correct digit.
So the MNIST data set has got
60,000 images of handwritten digits.
Here's a smattering of them.
And we can use this training set to learn a classifier.
A function that takes an image and then outputs
what digit it thinks it is.
MNIST also has a separate test set of 10,000 images
along with their labels, and we can use this test set
to assess how good our classifier really is.
So what kind of classifiers might we try?
Well, the simplest one imaginable perhaps,
is Nearest Neighbor.
What happens in this case is that
when we get a new image to classify,
say this one over here, we get some new image X.
We go through our training set of 60,000 images
and we find the one that's closest to X.
Then we simply return the label of that image.
That's it.
Now, there are some details that we have to work out.
If we are looking for the one that's closest to X,
it means we have some notion of distance between images.
How are we representing images on the computer anyway?
So let's look into that.
So first off, we will represent images as vectors, okay?
Now an MNIST image is 28 pixels by 28.
So it's 28 pixels across, 28 pixels high.
That means the total number of pixels
is 28 and 28 which is 784.
And each pixel is grayscale.
So it's a value in the range zero to 255
where zero means black and 255 means white.
For example, the pixels in the upper corner over here
are all zero, whereas some of the pixels
in the middle over here are probably much closer to 255.
What we'll do with an image like this
in order to make it into a vector is to
simply stretch it out into one long, 784 dimensional vector.
Something like this, okay?
And how do we stretch it out?
Well, we begin by just copying down the first row.
So we copy down the first row over here.
Then we copy down the second row.
And all the way to the last row.
So these initial positions are all zeros.
Somewhere in the middle, we have numbers like 200
and towards the end, we have zeros again.
So we've taken the image and converted it into
a 784 dimensional vector.
Our data space then, which we're gonna denote by script X
is 784 dimensional Euclidean space
and we'll often write it like this: R to the 784th.
The label space just consists of
the possible labels, zero to nine.
Now that we have a specific vector representation,
we also have to decide how we're going to compute distances
between vectors and the most common, or default distance
function is perhaps just Euclidean distance.
So let's recall how this works in two dimensions.
When you have two points, the Euclidean distance
between them is just the length of the line connecting them.
So it's the length of this line.
And what is that length?
Well, if you look at these two points, X and Z,
along the first coordinate, they defer by two
and along the second coordinate, they defer by three.
So the length of the line, the distance from X to Z
is simply the square root of two squared plus three squared
which is the square root of 13.
That's the Euclidean distance between X and Z
in two dimensions, okay?
Now of course we aren't working in two dimensions.
We're working in a much higher dimensional space
but the basic idea is the same.
When you want to compute the distance
between two vectors, X and Z, you simply find out
how much they differ on each individual coordinate,
you square these values, you add them up
and then you take the square root of the whole thing.
That's Euclidean distance.
Good.
So now we have a representation of the images
as vectors in 784 dimensional space
and we have a distance function between images.
So we're ready to use nearest neighbor.
Each time we get a new image,
we simply find its nearest neighbor
using Euclidean distance in 784 dimensional space
and we return the label of this training image.
So how good is this classifier?
Well, let's look at some numbers.
First of all, what is the error rate
of the classifier on the training points?
So we have these 60,000 training images.
For any training point, its nearest neighbor
in the training set is itself.
So it'll definitely get the right label.
So the error rate on the training set is zero.
What that means is that training error
is not a good predictor of future performance.
It in general is something that is overly optimistic.
That's why we have a separate test set.
If we compute the error on those separate 10,000 points,
that's really a much better indication
of how well this classifier is gonna perform in practice.
Now, what kind of test error might we expect?
Well, let's do a little tore experiment.
Suppose that we use the classifier
that was completely random.
When it was given an image, it didn't even look at the image
but just randomly chose a number from zero to nine.
What would be the error rate of a classifier like this?
Well, whatever the correct label is,
the chance that it randomly picks
that correct label is 10%.
So a random classifier has got an error rate of 90%.
We certainly want to do better than that.
Now let's see how well nearest neighbor does.
On the test set, its error rate is 3.09%.
That means that out of the 10,000 points,
it gets 309 of them wrong.
That's not too bad for such a simple method.
Let's look at some of the mistakes that it makes.
This query, for example.
When it was looking for its nearest neighbor,
this is the point it found.
So it thought it was a four.
Look at this one.
Its nearest neighbor in the training set
turned out to be this point and so it thought
it was an eight, and so on and so forth.
These errors are all quite understandable
once you see what the nearest neighbor classifier is doing.
So, we have our first classifier now.
And next time, we'll see how to make it better.

I included some posts for reference.

https://github.com/noey2020/How-to-Talk-Matlab-Tricks-and-Tweaks

https://github.com/noey2020/How-to-Talk-Trading-and-Investing

https://github.com/noey2020/How-to-Work-in-Matlab-Development-Environment

https://github.com/noey2020/How-to-Talk-Vaccines

https://github.com/noey2020/How-to-Talk-Regression-in-Matlab

https://github.com/noey2020/How-to-Get-Started-in-Matlab

https://github.com/noey2020/How-to-Convert-Data-from-Web-Service-Using-Matlab

https://github.com/noey2020/Quote-for-the-Day

https://github.com/noey2020/How-to-Talk-Good-Investment-Strategy

https://github.com/noey2020/How-to-Talk-of-Good-Plan

https://github.com/noey2020/Thought-for-the-Day

https://github.com/noey2020/How-to-Talk-Stock-Watch-of-the-Day

https://github.com/noey2020/How-to-Talk-Data-Science

https://github.com/noey2020/How-to-Talk-Fundamental-Analysis

https://github.com/noey2020/How-to-Read-Company-Profiles

https://github.com/noey2020/How-to-Import-Data-from-Spreadsheets-and-Text-Files-Matlab-Without-Coding

https://github.com/noey2020/How-to-Talk-Model-of-Stock-Market-Prices-

https://github.com/noey2020/How-to-Talk-Digital-Wallets

https://github.com/noey2020/How-to-Talk-Investing

https://github.com/noey2020/How-to-Double-Your-Money-in-5years

https://github.com/noey2020/How-to-Talk-Matlab

I appreciate comments. Shoot me an email at noel_s_cruz@yahoo.com!
