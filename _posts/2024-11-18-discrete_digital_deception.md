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

Let’s check with numbers. In the digital world, even basic math can be misleading. If you've ever tried to add `1.1` and `2.2` in Python (or any other programming language), you'll notice the result isn’t exactly `3.3`.

```python
1.1 + 2.2 == 3.3
```
**Output:**
```plaintext
False
```
![math-lady](/assets/img/digital-illusions/math-lady.jpg)


The digital world relies on the **IEEE 754 Standard for Floating-Point Arithmetic**, a method of representing fractions digitally. This system introduces rounding errors because computers can only store a finite number of decimal places. Think of it as trying to pour a gallon of water into a pint-sized jar.
Instead of dealing with precise values, computers approximate. The result? Slight inaccuracies like the one above. Computers use binary to represent numbers, and binary isn’t great at storing fractions exactly. So, what you’re seeing is the closest approximation.

Think of public transport, like a bus or train, with predetermined stops. Your destination might not match a stop exactly, so you get off at the nearest one and walk the rest of the way. Similarly, computers approximate numbers to the nearest value they can represent, just wish it’s a perfect match.

---


## How IEEE 754 Impacts Machine Learning

When you add $0.1$ three times, you might expect the result to be exactly $0.3$. However, due to floating-point approximations, the result deviates slightly.

```python
result = 0.1 + 0.1 + 0.1

print("Result of 0.1 + 0.1 + 0.1:", result)
print("Is result equal to 0.3?:", result == 0.3)
```

---

```plaintext
Result of 0.1 + 0.1 + 0.1: 0.30000000000000004
Is result equal to 0.3?: False
```

In machine learning and numerical computations, such errors can:
 - **Impact Equality Comparisons:** Direct comparisons (e.g., `a == b`) may fail due to tiny differences caused by floating-point errors.
   
 - **Break Algorithms:** Algorithms relying on precise equality checks (e.g., sorting, clustering) may behave unexpectedly.

### **How to Mitigate This?**

#### Use `math.isclose`:
Instead of direct equality, use a tolerance:
```python
import math

print("Is result approximately equal to 0.3?:", math.isclose(result, 0.3))
```

**Output:**
```plaintext
Is result approximately equal to 0.3?: True
```

---

#### Use the `decimal` Module:
Python’s `decimal` module provides precise arithmetic for situations where exact results are needed:
```python
from decimal import Decimal

result = Decimal('0.1') + Decimal('0.1') + Decimal('0.1')
print("Result with Decimal:", result)
print("Is result equal to 0.3?:", result == Decimal('0.3'))
```

**Output:**
```plaintext
Result with Decimal: 0.3
Is result equal to 0.3?: True
```

In machine learning, we frequently standardize data to scale features to a mean of `0` and a standard deviation of `1`. However, this process is not immune to the quirks of the **IEEE 754 floating-point standard**, which governs how numbers are represented in digital systems. These quirks can introduce precision errors that disrupt calculations, particularly in large-scale or small-scale datasets.

In machine learning workflows, these floating-point quirks can cause:
 - **Distorted Feature Scaling:** Features with extreme magnitudes (very large or small) lose accuracy during preprocessing.
 - **Poor Convergence in Models:** Gradient-based optimizers rely on precise calculations, and any error in feature scaling propagates during training.
 - **Amplified Noise in Sparse Data:** Sparse datasets may introduce unexpected biases due to exaggerated small values after standardization.

These issues manifest prominently during standardization due to the formula:

$$
z = \frac{x - \mu}{\sigma}
$$

Where:
- $x$ is the data point,
- $\mu$ is the mean,
- $\sigma$ is the standard deviation.

When $\mu$ or $\sigma$ is very large or very small, **floating-point precision errors** can cause distortion.

---

### Amplification of Precision Errors
If $\sigma$ (standard deviation) is very small or $\mu$ (mean) involves large numbers, significant digits may be lost during subtraction or division.

```python
import numpy as np

data = np.array([1e10 + 1e-5, 1e10 + 2e-5, 1e10 + 3e-5])
mean = np.mean(data)
std = np.std(data)

# Standardization
standardized = (data - mean) / std
print("Standardized Data:", standardized)
```

The mean of these values is:

$$
\mu = \frac{(1e10 + 1e-5) + (1e10 + 2e-5) + (1e10 + 3e-5)}{3} = 1e10 + 2e-5
$$

The standard deviation measures the spread of the data points. Since the values differ by equal increments (`1e-5`):

$$
\sigma = 1e-5
$$

For each data point, the standardized value is:

$$
z = \frac{x - \mu}{\sigma}
$$


Substituting the values:
- For `x = 1e10 + 1e-5`:
  $$
  z = \frac{(1e10 + 1e-5) - (1e10 + 2e-5)}{1e-5} = -1
  $$
- For `x = 1e10 + 2e-5`:
  $$
  z = \frac{(1e10 + 2e-5) - (1e10 + 2e-5)}{1e-5} = 0
  $$
- For `x = 1e10 + 3e-5`:
  $$
  z = \frac{(1e10 + 3e-5) - (1e10 + 2e-5)}{1e-5} = 1
  $$


Thus, you would **ideally** expect the output to be:
```plaintext
Standardized Data: [-1, 0, 1]
```

Well, run the code yourself and be amazed to see the output as:
```plaintext
Standardized Data: [-1.31982404 -0.21997067  1.09985336]
```

Due to the **IEEE 754 floating-point standard**, large numbers like `1e10` lose precision when subtracted from similar large numbers (e.g., `1e10 - 1e10`). This phenomenon, called **catastrophic cancellation**, causes the computation to lose significant digits, introducing inaccuracies into the standardized result.

---

### Errors in Sparse or Noisy Data
Sparse datasets with very small values (`1e-9, 1e-10`) are particularly prone to precision issues. Standardizing these values often magnifies errors.

```python
data = np.array([1e-9, 2e-9, 3e-9])
mean = np.mean(data)
std = np.std(data)

standardized = (data - mean) / std
print("Standardized Data:", standardized)
```

The tiny differences between the values will appear exaggerated after standardization. This behavior arises because floating-point representation cannot accurately handle extremely small values.

When numbers are extremely small, rounding errors dominate, making the transformation potentially misleading.

---


Understanding these limitations helps us make better decisions when working with digital data, ensuring our machine learning models remain robust and reliable.

---


## Final Thoughts: The Future of Illusions
The digital world is all about tricking your senses, and it’s only going to get better at it. With AI, hyper-realistic graphics, and 3D audio, we’re headed toward a future where it’ll be nearly impossible to distinguish real from digital. While this is exciting, it also makes you wonder: how much of our perception is based on what’s real, and how much is just clever engineering?

As Albert Einstein said,*"The human mind has first to construct forms, independently, before we can find them in things"* highlights how perception is not passive but an active construction process, susceptible to errors and illusions.

The next time you watch a movie, listen to music, or even do some math on your computer, remember: it’s all discrete, it’s all outward appearance, but wow, does it work.

![this-is-fine-dog](/assets/img/digital-illusions/this-is-fine-dog.jpg)

---
