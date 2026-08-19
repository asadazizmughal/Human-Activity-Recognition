# Findings: Human Activity Recognition from Accelerometer Data

This document explains the whole project in one place: what the goal was, how the
data was handled, what each stage found, and what the results actually mean. It is
written to stand on its own, so you do not need to open the notebooks to follow it.
The notebooks hold the code and the plots; this is the reasoning behind them.

## 1. What the project is and why it was built

The task is to recognise six everyday activities (walking, walking upstairs,
walking downstairs, sitting, standing, laying) from a wearable inertial sensor.
The sensor records acceleration and rotation, and the model has to decide which
activity produced a short window of that signal.

The reason for building it this way comes from earlier research work at EuroMov,
where the problem was predicting real-world arm use in stroke patients from a
wrist-worn accelerometer. That work made one lesson very clear. With sensor data
from people, you cannot let the same person appear in both the training data and
the test data. If you do, the model can memorise that person instead of learning
the activity, and your reported accuracy stops meaning anything. This project
takes that lesson and applies it to a clean public benchmark, so the method can be
shown openly and reproduced by anyone.

So the point of the project is not to chase the highest possible accuracy number.
The point is to measure honestly, on people the model has never seen, and to
explain the results rather than just report them.

## 2. The data

The project uses the UCI HAR dataset. Thirty volunteers wore a waist-mounted
smartphone. The phone recorded accelerometer and gyroscope signals at 50 samples
per second. The continuous recording was cut into windows of 2.56 seconds, which
is 128 samples each, with the windows overlapping by half.

The dataset comes in two forms, and this project uses both:

- **Engineered features.** Each window is already summarised into 561 numbers
  (means, standard deviations, frequency measures, and so on). This is the ready
  made form most people use.
- **Raw signals.** The same windows, but as the raw sensor readings: 128 timesteps
  across 9 channels (three axes each for body acceleration, gyroscope, and total
  acceleration). This is the harder form, where the model has to work out for
  itself what matters.

The split between training and test data is the important part. Of the 30 people,
21 are placed in the training set (7,352 windows) and 9 are placed in the test set
(2,947 windows). No person is in both. Every accuracy number in this project is
measured on those 9 held-out people, so it reflects how the model does on someone
new, which is the situation that matters in the real world.

## 3. The method that ties everything together

The single idea running through all four stages is subject-independent evaluation.
It is worth spelling out because it is easy to get wrong, and getting it wrong is
the most common flaw in activity-recognition tutorials.

The wrong way is to take all the windows from all the people, shuffle them, and
split randomly into train and test. This looks fine, but it lets windows from the
same person land on both sides. Because two windows from the same person doing the
same activity look very similar, the model can score highly just by recognising
the person. Accuracy on that kind of split is often above 98 percent, and it is
misleading.

The right way, used here, keeps whole people on one side or the other. The model
trains on 21 people and is tested on 9 different people. When it is later asked to
hold out part of the training data for tuning (in the deep learning stage), that
holdout is also done by person, never by window. The discipline is the same at
every level. This is the reason the results in this project are a few points lower
than the inflated numbers you often see, and it is also the reason they can be
trusted.

## 4. What the data itself showed (before any modelling)

Before training anything, the raw data was examined to understand what a model
would be up against.

**The classes are mildly imbalanced.** The static activities and laying have
somewhat more windows than the three walking activities. The imbalance is not
severe, but it is enough that plain accuracy can flatter a model. For that reason
the main score used throughout is the macro-averaged F1 score, which treats every
activity as equally important regardless of how many windows it has.

**Moving and still are easy to separate.** Plotting the acceleration for one window
of each activity shows the intuition immediately. The three walking activities
oscillate in a regular rhythm, because walking is a repeating gait cycle. Sitting,
standing, and laying are close to flat lines, because almost nothing moves. So
telling a moving activity from a still one is not the hard part.

**Sitting and standing are the hard case, and the reason is physical.** The raw
signal splits into two pieces: body acceleration, which is the movement with
gravity removed, and total acceleration, which includes gravity. Looking at body
acceleration, sitting and standing are almost identical, because neither involves
real movement. The only thing that separates them is orientation, the direction of
gravity, which shows up in the total-acceleration signal and shifts slightly
depending on posture. That cue is subtle, and it changes from person to person
depending on how they sit or how the phone is angled. So a model tested on new
people has very little reliable information to tell sitting from standing. This
prediction, made purely from looking at the signal, is what the later confusion
matrices confirm.

**A two-dimensional map of the features agrees.** Compressing the 561 features down
to two dimensions with PCA produces a picture where the moving activities spread
out on one side and the still activities bunch together on the other, with the
still ones overlapping. The easy separation and the hard separation are both
visible in a single plot.

## 5. Understanding the features by rebuilding them

The dataset hands over 561 features, and it would be easy to use them without ever
knowing what they are. To avoid that, a smaller set of features was built by hand
directly from the raw signals. For each of the 9 channels, the code computes
time-domain measures (mean, standard deviation, minimum, maximum, energy,
mean-crossing rate, skewness, kurtosis, and a few more) and frequency-domain
measures (dominant frequency, spectral energy, spectral entropy, spectral
centroid), plus correlations between the axes of each sensor. This gives 162
features in total.

The test was simple: train the same model on the hand-built 162 features and on
the official 561, and compare on the held-out people. The hand-built set reached
0.921 macro-F1 against 0.926 for the official set. In other words, a set less than
a third the size, built from scratch, came within about half a percent of the
official one. The lesson is that the 561 features are not magic. They are more of
the same kinds of statistics, and once you can build a competitive set yourself,
you understand what the model is actually working with.

Looking at which of the hand-built features the model leaned on most, the top ones
split cleanly into the two jobs the EDA pointed to. Total-acceleration magnitude
measures (max, root-mean-square, energy) capture how much movement there is, which
separates moving from still. Total-acceleration mean and median capture
orientation, which is the weak cue that separates the still postures from each
other. The features the model finds useful match the physical explanation.

## 6. Classical models and the honest result

Four classical models were trained on the official 561 features and tested on the
9 held-out people: Logistic Regression as a simple linear baseline, Random Forest,
Gradient Boosting, and a Support Vector Machine with an RBF kernel. Linear and
kernel models had their features scaled first, with the scaler fit on training
data only so that no test information leaked in.

The results, sorted by macro-F1:

| Model               | Accuracy | Macro-F1 | Fit time |
|---------------------|----------|----------|----------|
| Logistic Regression | 0.955    | 0.955    | about 2 s |
| SVM (RBF)           | 0.955    | 0.954    | about 2 s |
| Gradient Boosting   | 0.933    | 0.933    | about 34 s |
| Random Forest       | 0.926    | 0.924    | about 34 s |

The first thing to notice is that the simple linear model is both the most
accurate and the fastest. That is not an accident. The 561 features were designed
by the dataset authors specifically to make the activities close to linearly
separable, so a linear model already does almost the whole job. The heavier
ensembles, which take about fifteen times longer to train, buy nothing on top.
This is a useful result to be able to explain: reaching for the most complex model
is not always right, and here the simplest one wins.

The second thing is the confusion matrix. Every model, without exception,
concentrates its mistakes in the same place: sitting predicted as standing. And
the error is not symmetric. About one sitting window in eight is labelled as
standing, while standing is mislabelled as sitting only rarely. In the per-class
numbers this shows up as sitting having the lowest recall (many sitting windows
are missed) and standing having the lowest precision (it absorbs those misread
sitting windows). This is exactly what the EDA predicted from the physics of the
signal. Nothing else is confused to any meaningful degree.

This is the payoff of the subject-independent method. On a leaky random split this
error would have been buried under an accuracy figure near perfect. Here it shows
up plainly, and it lines up with a physical explanation, which is what gives it
credibility.

## 7. Deep learning on the raw signal

The last stage asked a different question. The classical models worked on 561
hand-engineered features. Could a model given only the raw signal, and left to
work out its own features, do as well or better?

Two deep models were trained on the raw 128-by-9 windows. A one-dimensional
convolutional network slides filters across the time axis to learn short motion
patterns, then pools them into a summary. A bidirectional LSTM reads the 128
timesteps forwards and backwards. Both were trained with a validation set held out
by person, not by window, and training stopped early once the validation score
stopped improving.

Adding them to the table:

| Model               | Accuracy | Macro-F1 |
|---------------------|----------|----------|
| Logistic Regression | 0.955    | 0.955    |
| SVM (RBF)           | 0.955    | 0.954    |
| Gradient Boosting   | 0.933    | 0.933    |
| 1D CNN              | 0.924    | 0.925    |
| Random Forest       | 0.926    | 0.924    |
| BiLSTM              | 0.897    | 0.898    |

(Deep-learning numbers vary by a point or two from run to run and across hardware,
because training involves randomness. The pattern is stable even if the exact
figures shift.)

The CNN, learning entirely from raw signal, reaches the level of the tree
ensembles. The BiLSTM does a little worse. Neither beats the simple linear model.
This is the expected outcome on a dataset this small. Deep learning tends to win
when there is a lot of data and when good features are hard to design by hand.
Here there is not much data, and the features have already been designed very well,
so learning from scratch has almost no room to improve on them, and it costs far
more compute to try.

The more interesting result is the confusion matrix. The CNN, a completely
different kind of model built on completely different inputs, makes the same
mistake as the classical models: sitting confused with standing, even more
strongly than before. Two model families that share nothing in common agree on
what is easy and what is hard. That agreement is strong evidence that the limit
here is set by the sensor and the task, not by the choice of model. No amount of
extra modelling is going to fix an error that comes from the data itself.

## 8. What to take away

- On unseen people, a simple linear model on well-engineered features is the
  strongest and cheapest option for this task. Complexity did not help.
- Hand-built features reached within half a percent of the official 561, which
  shows the official features are understandable rather than magic.
- Every model, classical or deep, makes the same sitting-versus-standing error,
  and that error traces directly to the physics of the signal. It is a property of
  the data.
- All of this is only visible because the evaluation is subject-independent. That
  single choice is what separates a trustworthy result from an inflated one.

## 9. Honest limitations

- The dataset is small and was collected in a controlled setting with one sensor
  position. Results on messier, real-world data would likely be lower.
- The deep models were kept deliberately compact and were not heavily tuned. With
  far more data they might pull ahead, but that was not the situation here.
- Only the built-in train and test split was used for the headline numbers. A
  leave-one-subject-out cross-validation would give a more complete picture of how
  much the score varies from person to person, and is a natural next step.

## 10. Possible next steps

- Run leave-one-subject-out cross-validation to measure per-person variation.
- Add a second dataset with a different sensor placement to test whether the same
  pipeline holds up across setups.
- Look specifically at the sitting-versus-standing boundary, for example by adding
  orientation-focused features, to see how far that specific error can be reduced.
