---
title: "Discrete Digital Deception: Bits Fooling You"
description: "How tech plays the game, making fake look the same."
author: supreeth
date: 2024-11-18 10:00:00 +0800
last_modified_at: 2024-11-18 10:00:00 +0800
categories: [Fundamentals, Machine Learning]
tags: [Machine Learning, Data Science, Digital Content, Discrete Math]
math: true
image:
  path: /assets/img/digital-illusions/digital-matrix.webp
  lqip: data:image/jpeg;base64,UklGRkYAAABXRUJQVlA4IDoAAADQAQCdASoKAAYAAUAmJQBOgBukcAB0AAD++pm7jN5HKehjoEulPc5Ob4IoBipGpT89Vvi2W7bumAAA
  alt: Everything we know is only some kind of approximation, because we do not yet know all the laws of nature. — Richard P. Feynman
---


## Introduction: Welcome to the Matrix!
What’s your screen time today?

We’re living in an increasingly digital world, where nothing is truly continuous. Everything you see, hear, and experience digitally is an approximation, a clever lie designed to trick your senses. Let’s dive into the digital world, where reality is cut into tiny, discrete pieces to fit the rigid framework of 1s and 0s. 

From how your computer adds numbers to how your favorite song streams, everything in the digital realm is a deception orchestrated to satisfy our limited human perception. Let’s dive into this fascinating world and uncover the tricks of the digital trade.

---

## Examples of Everyday Digital Trickery

---

### Lights, Camera, Illusion!

Let’s talk about videos. The smoothness of a video is an illusion created by displaying a sequence of still images (frames) rapidly. At 120 frames per second (fps), the video presents a sequence of still images so rapidly that your brain perceives them as smooth motion. Our eyes can’t detect the individual frames beyond a certain threshold, usually around 24–60 fps for most humans. However, your pet might not share this experience because animals like dogs and cats have higher flicker fusion thresholds, meaning they perceive the gaps between frames more easily and see the screen as flickering chaos.

![huh-dog](/assets/img/digital-illusions/huh-dog.gif)


But the illusion isn’t just about the fps of the video. Irrespective of whether you are watching a video or reading this page, the screen also has a refresh rate measured in hertz (Hz). This determines how many times per second the screen updates the displayed content. For example, a screen with a 60Hz refresh rate updates 60 times per second, while a 120Hz screen updates twice as fast. If the fps of the video exceeds the refresh rate of the screen, those extra frames are either skipped or blended, which might degrade the smoothness of motion. Conversely, if the screen’s refresh rate is higher than the fps of the video, it introduces smoother motion, but only up to the limitations of the video content.

In essence, fps is how the software delivers the frames, while refresh rate is how the hardware displays them. For the smoothest experience, they need to work in harmony.

Even a single frame in that video isn’t “continuous” like reality. What looks real to you is just an arrangement of tiny pixels. More pixels = higher resolution = sharper illusion. But zoom in on any pixel, and you realize it’s just a tiny blob of color, hardly the Mona Lisa!

Think of pixels like LEGO bricks. A low-res image is like using chunky blocks to build a castle. It gets the idea across but looks rough. A high-resolution image is like using tiny LEGO pieces for presenting subtle details. Either way, it’s still just LEGO!


![wojak-and-chad](/assets/img/digital-illusions/wojak-and-chad.jpg)

---

### Your Ears Are Just as Gullible
Did you know your favorite song is just numbers? Yep, digital audio records sound by taking snapshots of it, typically 44,000 times per second (44.1 kHz). Each snapshot is a single value, and when played back in sequence, it tricks your ears into hearing a smooth melody.

Here’s the twist: even at a live concert with analog instruments, the sound you hear isn’t entirely *pure*. For a large audience, microphones capture the sound waves, convert them into electrical signals, and loudspeakers recreate them. While the final sound mimics the real thing, it remains an interpretation—one step removed from nature’s true analog form.

---

### The Math: When 1.1 + 2.2 ≠ 3.3

Let’s start with numbers. In the digital world, even basic math can be misleading. If you've ever tried to add `1.1` and `2.2` in Python (or any other programming language), you'll notice the result isn’t exactly `3.3`.


```python
>>> 1.1 + 2.2 == 3.3
False
```
Because

```python
>>> 1.1 + 2.2
3.3000000000000003
```

The digital world relies on the **IEEE 754 Standard for Floating-Point Arithmetic**, a method of representing fractions digitally. This system introduces rounding errors because computers can only store a finite number of decimal places. Think of it as trying to pour a gallon of water into a pint-sized jar.
Instead of dealing with precise values, computers approximate. The result? Slight inaccuracies like the one above. Computers use binary to represent numbers, and binary isn’t great at storing fractions exactly. So, what you’re seeing is the closest approximation.

Think of public transport, like a bus or train, with predetermined stops. Your destination might not match a stop exactly, so you get off at the nearest one and walk the rest of the way. Similarly, computers approximate numbers to the nearest value they can represent, just wish it’s a perfect match.

![math-lady](/assets/img/digital-illusions/math-lady.jpg)

---


## Adverse Effects in Machine Learning

### The Effect of IEEE 754 in Standardization

When performing operations like standardization, values are transformed using the formula:

$$
z = \frac{x - \mu}{\sigma}
$$

Here:
- $x$ is the data point,
- $\mu$ is the mean of the feature,
- $\sigma$ is the standard deviation.

If $x$, $\mu$, or $\sigma$ have very small or very large magnitudes, floating-point approximations can affect the computation, especially when $x$ is close to $\mu$.

**Example of Precision Loss:**
```python
import numpy as np

# Simulate data with small differences
data = np.array([1.0000001, 1.0000002, 1.0000003])
mean = np.mean(data)
std = np.std(data)

# Standardize
standardized = (data - mean) / std
print("Standardized Data:", standardized)
```

**Output:**
```
Standardized Data: [-1.22474487, 0.0, 1.22474487]
```

Notice how despite the tiny differences in the original data, the output seems exaggerated. Floating-point precision amplifies these differences unnaturally.

---


### Loss of Information for Sparse Data
If a feature has many zeros or near-constant values, standardization can amplify noise or lead to values close to machine epsilon (~ $10^{-16}$), which the model may misinterpret as significant.

**Example:**
```python
sparse_data = np.array([0, 0, 0, 1e-10, 1e-9, 0])
mean = np.mean(sparse_data)
std = np.std(sparse_data)

standardized = (sparse_data - mean) / std
print("Standardized Sparse Data:", standardized)
```

**Output:**
```
Standardized Sparse Data: [-0.57735027 -0.57735027 -0.57735027  0.11547005  1.03890377 -0.57735027]
```

The standardized values are distorted due to the small magnitudes of the original data.

**Use Case:**  
Models relying on such features (e.g., logistic regression) may struggle to learn meaningful patterns, especially for imbalanced datasets.

---

### Impact on Gradient-Based Optimizers
Standardization introduces numerical instability when gradients involve very small or very large scaled values. This can affect convergence during training.

**Example with Neural Networks:**
```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

# Simulate poorly scaled features
X = np.array([[1e6, 1e-6], [1e6 + 1, 1e-6 + 1e-9], [1e6 - 1, 1e-6 - 1e-9]])
y = [0, 1, 0]

# Without standardization
clf = SGDClassifier(loss="log")
clf.fit(X, y)
print("Without Standardization:", clf.coef_)

# With standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
clf.fit(X_scaled, y)
print("With Standardization:", clf.coef_)
```

**Output:**  
Without standardization, the model struggles to handle the scale disparity. With standardization, it performs better but risks introducing bias from numerical instability.

---

## When to Be Cautious

### 1. Very Small Standard Deviations
When features have nearly constant values, dividing by a tiny standard deviation amplifies noise.

**Mitigation:** Add a small epsilon ($\epsilon$) to the denominator:
```python
epsilon = 1e-8
standardized = (data - mean) / (std + epsilon)
```

### 2. Outliers
Extreme values disproportionately influence the mean and standard deviation, skewing the transformed values.

**Mitigation:** Use robust scaling methods, such as scaling based on the median and interquartile range.

---


## Final Thoughts: The Future of Illusions
The digital world is all about tricking your senses, and it’s only going to get better at it. With AI, hyper-realistic graphics, and 3D audio, we’re headed toward a future where it’ll be nearly impossible to distinguish real from digital. While this is exciting, it also makes you wonder: how much of our perception is based on what’s real, and how much is just clever engineering?

As Albert Einstein said,*"The human mind has first to construct forms, independently, before we can find them in things"* highlights how perception is not passive but an active construction process, susceptible to errors and illusions.

The next time you watch a movie, listen to music, or even do some math on your computer, remember: it’s all discrete, it’s all outward appearance, but wow, does it work.

![this-is-fine-dog](/assets/img/digital-illusions/this-is-fine-dog.jpg)

---
