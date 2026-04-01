# Knee Arthritis Detective! 🕵️‍♀️🦵

Hello there! Welcome to our super cool project. 

## What does this do? 
Imagine you have a magic magnifying glass that can look at pictures of knees (X-rays) and tell you if they are sick and how sick they are. This project is exactly that! It uses computer brains (AI) to look at knee X-rays and find out if a person has "knee arthritis" (which is when the knee starts to hurt because the cushion inside is wearing out over time).

## How does the magic work? 🎩✨
Our project has a super team of **3 different computer detectives** plus **1 final boss detective**. Let's meet them!

1. **Detective 1 & Detective 2**: These two look at the whole knee picture very closely to figure out the "Severity Grade". They try to guess how bad the knee is on a scale from 0 to 4: 
   * 😊 Normal
   * 🤔 Doubtful
   * 😕 Mild
   * 😟 Moderate
   * 😭 Severe
2. **Detective 3**: This detective looks for very specific clues, like extra tiny bone pieces or worn-out spots (we call this "Morphology").
3. **The Boss Detective (Random Forest)**: This is our smartest computer brain! It doesn't look at the picture directly. Instead, it listens to the other three detectives, takes all their guesses, thinks really hard, and makes the final, super-accurate decision on how the knee is doing!

## The Two Main Files 📁
You will see two big superhero files doing all the heavy lifting:

* `meta_fuse.py` (The School 🏫): This is where we send our Boss Detective to school! It downloads thousands of knee pictures from the internet, asks the three detectives to practice on them, and trains the Boss Detective to make the best decisions.
* `meta_eval_rf.py` (The Real Test 🩺): This is the script you use when you want to test a real knee! You give it one picture of a knee. It quickly downloads our pre-trained detectives from the cloud (Google Drive), shows them the picture, and then prints out a neat report card telling you the final answer and how confident they are!

## How to play? 🎮
You can use `meta_eval_rf.py` to test it out! It will show you exactly what the detectives think about a knee X-ray. It even draws cool loading bars using `#` to show you how sure it is!

Stay curious and happy coding! 🚀
